from io import BytesIO
from pathlib import Path
import time

import numpy as np
import pandas as pd
from PIL import Image
import torch
from transformers import CLIPModel, CLIPTokenizer

from data_loader import MEAN, STD, SIZE, make_split, read_aokvqa

CLIP = "openai/clip-vit-base-patch32"
OUT = Path(__file__).resolve().parent.parent / "data" / "aokvqa" / "features"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH = 128

def image_tensor(image_bytes):
    # same preprocessing as the data loader: resize to 224 x 224, CLIP normalisation
    image = Image.open(BytesIO(image_bytes)).convert("RGB").resize((SIZE, SIZE), Image.BICUBIC)
    x = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255
    return (x - torch.tensor(MEAN).view(3, 1, 1)) / torch.tensor(STD).view(3, 1, 1)

@torch.inference_mode()
def encode_images(model, images):
    # global vector (512), last tokens (50 x 768) and the input of the last layer (for the fine-tune ablation)
    glob, tokens, before_last = [], [], []
    for start in range(0, len(images), BATCH):
        x = torch.stack([image_tensor(b) for b in images[start:start + BATCH]]).to(DEVICE)
        out = model.vision_model(pixel_values=x, output_hidden_states=True)
        glob.append(model.visual_projection(out.pooler_output).half().cpu())
        tokens.append(out.last_hidden_state.half().cpu())
        before_last.append(out.hidden_states[-2].half().cpu())
    return torch.cat(glob), torch.cat(tokens), torch.cat(before_last)

@torch.inference_mode()
def encode_texts(model, tokenizer, texts, max_len):
    # global vector (512) and the token states (max_len x 512) with their mask
    glob, tokens, masks = [], [], []
    for start in range(0, len(texts), BATCH):
        batch = tokenizer(texts[start:start + BATCH], padding="max_length", truncation=True,
                          max_length=max_len, return_tensors="pt").to(DEVICE)
        out = model.text_model(**batch)
        glob.append(model.text_projection(out.pooler_output).half().cpu())
        tokens.append(out.last_hidden_state.half().cpu())
        masks.append(batch["attention_mask"].bool().cpu())
    return torch.cat(glob), torch.cat(tokens), torch.cat(masks)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = read_aokvqa()
    manifest = make_split(df)
    df = df.merge(manifest[["question_id", "split"]], on="question_id")
    model = CLIPModel.from_pretrained(CLIP).to(DEVICE).eval()
    tokenizer = CLIPTokenizer.from_pretrained(CLIP)

    # one row per distinct image, questions point to their image by index
    keys = list(dict.fromkeys(df["image_key"]))
    first = df.drop_duplicates("image_key").set_index("image_key")
    start = time.perf_counter()
    v, v_tokens, v_before = encode_images(model, [first.loc[k, "image"] for k in keys])
    image_seconds = time.perf_counter() - start

    questions = list(df["question"])
    choices = [list(c) for c in df["choices"]]
    pairs = ["%s answer: %s" % (q, c) for q, cs in zip(questions, choices) for c in cs]
    lengths = [len(tokenizer(t)["input_ids"]) for t in pairs]
    max_len = min(77, max(lengths))
    start = time.perf_counter()
    q, _, _ = encode_texts(model, tokenizer, questions, max_len)
    c, _, _ = encode_texts(model, tokenizer, [x for cs in choices for x in cs], max_len)
    qc, qc_tokens, qc_mask = encode_texts(model, tokenizer, pairs, max_len)
    text_seconds = time.perf_counter() - start

    n = len(df)
    meta = pd.DataFrame({
        "question_id": df["question_id"].values, "split": df["split"].values,
        "label": df["correct_choice_idx"].astype(int).values,
        "image_index": [keys.index(k) for k in df["image_key"]],
        "question": questions, "choices": choices,
        "direct_answers": list(df["direct_answers"]), "difficult": df["difficult_direct_answer"].astype(bool).values,
    })
    meta.to_pickle(OUT / "meta.pkl")
    torch.save({"v": v, "v_tokens": v_tokens, "v_before_last": v_before}, OUT / "image.pt")
    torch.save({"q": q, "c": c.view(n, 4, -1), "qc": qc.view(n, 4, -1),
                "qc_tokens": qc_tokens.view(n, 4, max_len, -1), "qc_mask": qc_mask.view(n, 4, max_len)}, OUT / "text.pt")
    print("questions %d | images %d | max tokens %d" % (n, len(keys), max_len))
    print("image encoding %.1f s | text encoding %.1f s | device %s" % (image_seconds, text_seconds, DEVICE))

if __name__ == "__main__":
    main()
