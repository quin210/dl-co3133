import torch
from torch import nn
from transformers import CLIPModel

CLIP = "openai/clip-vit-base-patch32"
D = 512

def mlp(inputs, hidden=512, dropout=0.1):
    # one hidden layer, gives one score per choice
    return nn.Sequential(nn.Linear(inputs, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

def unit(x):
    # CLIP vectors are compared by direction, so scale them to length 1
    return nn.functional.normalize(x.float(), dim=-1)

class TextOnly(nn.Module):
    # modality A: question and choice, no image
    def __init__(self):
        super().__init__()
        self.head = mlp(3 * D)

    def forward(self, batch):
        q = unit(batch["q"]).unsqueeze(1).expand(-1, 4, -1)      # [B,4,512]
        c = unit(batch["c"])                                       # [B,4,512]
        return self.head(torch.cat([q, c, q * c], -1)).squeeze(-1)  # [B,4]

class ImageOnly(nn.Module):
    # modality B: image and choice, the question is removed
    def __init__(self):
        super().__init__()
        self.head = mlp(3 * D)

    def forward(self, batch):
        v = unit(batch["v"]).unsqueeze(1).expand(-1, 4, -1)
        c = unit(batch["c"])
        return self.head(torch.cat([v, c, v * c], -1)).squeeze(-1)

class LateFusion(nn.Module):
    # simple fusion: concatenate the global vectors of image, question and choice
    def __init__(self, products=True):
        super().__init__()
        self.products = products
        self.head = mlp((5 if products else 3) * D)

    def forward(self, batch):
        v = unit(batch["v"]).unsqueeze(1).expand(-1, 4, -1)
        q = unit(batch["q"]).unsqueeze(1).expand(-1, 4, -1)
        c = unit(batch["c"])
        parts = [v, q, c, v * c, q * c] if self.products else [v, q, c]
        return self.head(torch.cat(parts, -1)).squeeze(-1)

class CrossAttention(nn.Module):
    # improved fusion: the late-fusion features plus a region-level branch, where the tokens of
    # "question answer: choice" attend to the 50 image tokens
    def __init__(self, heads=8):
        super().__init__()
        self.text_norm = nn.LayerNorm(D)
        self.image_norm = nn.LayerNorm(768)
        self.image_proj = nn.Linear(768, D)
        self.attention = nn.MultiheadAttention(D, heads, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(D)
        self.head = mlp(7 * D)

    def forward(self, batch):
        tokens, mask = batch["qc_tokens"].float(), batch["qc_mask"]   # [B,4,L,512], [B,4,L]
        b, k, length, _ = tokens.shape
        image = self.image_proj(self.image_norm(batch["v_tokens"].float()))  # [B,50,512]
        image = image.repeat_interleave(k, 0)                                 # [B*4,50,512]
        queries = self.text_norm(tokens.reshape(b * k, length, D))           # [B*4,L,512]
        attended, _ = self.attention(queries, image, image, need_weights=False)
        attended = self.norm(queries + attended)
        keep = mask.reshape(b * k, length, 1).float()
        region = unit(((attended * keep).sum(1) / keep.sum(1)).view(b, k, D))  # [B,4,512]
        v = unit(batch["v"]).unsqueeze(1).expand(-1, k, -1)
        q = unit(batch["q"]).unsqueeze(1).expand(-1, k, -1)
        c = unit(batch["c"])
        qc = unit(batch["qc"])
        parts = [v, q, c, v * c, q * c, region, region * qc]
        return self.head(torch.cat(parts, -1)).squeeze(-1)

class LastBlock(nn.Module):
    # the last CLIP vision block, trained in the fine-tune ablation; it turns the cached
    # input of the last block into the image tokens and the global image vector
    def __init__(self):
        super().__init__()
        clip = CLIPModel.from_pretrained(CLIP)
        self.layer = clip.vision_model.encoder.layers[-1]
        self.post_layernorm = clip.vision_model.post_layernorm
        self.projection = clip.visual_projection

    def forward(self, before_last):
        tokens = self.layer(before_last.float(), None)
        tokens = tokens[0] if isinstance(tokens, tuple) else tokens
        return tokens, self.projection(self.post_layernorm(tokens[:, 0]))

class FineTuned(nn.Module):
    # a fusion head plus a trainable last CLIP block
    def __init__(self, head):
        super().__init__()
        self.block = LastBlock()
        self.fusion = head

    def forward(self, batch):
        tokens, v = self.block(batch["v_before_last"])
        return self.fusion(dict(batch, v_tokens=tokens, v=v))

def build(name):
    if name == "TextOnly":
        return TextOnly()
    if name == "ImageOnly":
        return ImageOnly()
    if name == "LateFusion":
        return LateFusion()
    if name == "LateFusionNoProduct":
        return LateFusion(products=False)
    if name == "CrossAttention":
        return CrossAttention()
    if name == "LateFusionFineTune":
        return FineTuned(LateFusion())
    if name == "CrossAttentionFineTune":
        return FineTuned(CrossAttention())
    raise ValueError("unknown model %r" % name)
