# CO3133 — Deep Learning and Its Applications (Semester-261)

Course project repository.
Website: https://quin210.github.io/dl-co3133/

Ho Chi Minh City University of Technology (HCMUT), VNU-HCM · Faculty of Computer Science and Engineering
Instructor: Lê Thành Sách

## Members

| Full name | Student ID | Main role | GitHub |
|---|---|---|---|
| Võ Thị Như Quỳnh | 2353041 | — | [quin210](https://github.com/quin210) |
| Mai Trung Kiên | 2352641 | — | [MaiTrungKien](https://github.com/MaiTrungKien) |
| Hứa Tuệ Minh | 2450018 | — | [HuaTueMinh](https://github.com/HuaTueMinh) |

## Contents

| Assignment | Topic | Weight |
|---|---|---|
| A1 | Foundations of Deep Learning Pipelines and Architectures | 40% |
| A2 | Deep Learning on Large-Scale Data and Specialized Tasks | 30% |
| A3 | Multimodal Deep Learning | 30% |

## Installation

```bash
conda create -n dl-co3133 python=3.11 -y
conda activate dl-co3133
pip install -r requirements.txt
```

## Dataset preparation

```bash
python -m src.data.download --dataset fashion_mnist --root data/
```

Fashion-MNIST is used for the main reported results. MNIST is used for development and
debugging only. CIFAR-10 is an optional extension.

## Train

```bash
python -m src.train --config configs/a1_linear.yaml
```

## Evaluate

```bash
python -m src.evaluate --config configs/a1_linear.yaml --ckpt experiments/<run_id>/best.pt
```

## Reproducibility

Every reported number is traceable to: config + data split + checkpoint + experiment ID + git tag.
Each submission is tagged accordingly (`a1-m1`, `a1-m2`, `a2-m1`, ...).

- Default seed: 42, set for Python, NumPy, PyTorch and cuDNN deterministic mode — see `configs/base.yaml`
- Split: fixed stratified train/validation/test split derived from the seed — see `src/data/`
- Checkpoint selection rule: highest validation macro-F1
- All models in the main comparison share the same split, seed and evaluation protocol

## Hardware and environment

- GPU: NVIDIA RTX 6000 Ada Generation, 46 GB
- Driver 555.42.02 · CUDA 12.5 · PyTorch cu124 wheels
- Python 3.11 · exact library versions in `requirements.txt`

## Links

- Reports: `reports/`
- Assignment pages: https://quin210.github.io/dl-co3133/
- AI usage disclosure: [AI_USAGE.md](AI_USAGE.md)
