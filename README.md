# CO3133 — Deep Learning and Its Applications (Semester-261)

Course project repository — **Group G-02**.
Website: https://quin210.github.io/dl-co3133/

Ho Chi Minh City University of Technology (HCMUT), VNU-HCM · Faculty of Computer Science and Engineering
Instructor: Lê Thành Sách

## Members

| Full name | Student ID | Main role | GitHub |
|---|---|---|---|
| Võ Thị Như Quỳnh | 2353041 | Pipeline configuration, repository and website | [quin210](https://github.com/quin210) |
| Mai Trung Kiên | 2352641 | Model training, experiment runs and result assets | [MaiTrungKien](https://github.com/MaiTrungKien) |
| Hứa Tuệ Minh | 2450018 | EDA notebooks and report figures | [HuaTueMinh](https://github.com/HuaTueMinh) |

## Contents

| Assignment | Topic | Weight |
|---|---|---|
| A1 | Foundations of Deep Learning Pipelines and Architectures | 40% |
| A2 | Deep Learning on Large-Scale Data and Specialized Tasks | 30% |
| A3 | Multimodal Deep Learning | 30% |


## Reproducing Assignment 2 (tomato leaf disease classification)

Report: `docs/reports/A2_M3.pdf` · Slides: `docs/reports/A2_M3_slides.pdf` · Assignment page: https://quin210.github.io/dl-co3133/assignment2.html

### Install

```bash
conda create -n dl-co3133 python=3.12      # numpy 2.5 needs Python >= 3.12
conda activate dl-co3133
pip install -r requirements.txt
```

### Data

```bash
cd Assignment2
python download.py      # 17,996 images from Hugging Face shreemunale/tomato-leaves-dataset, revision c029791 -> data/tomato/raw/
python data_loader.py   # resizes every image to 224 x 224 once -> data/tomato/images_224.npz (2.7 GB), prints one batch shape
```

The published train / validation / test split (14,404 / 1,796 / 1,796) is used as is.

### Train and evaluate

Every run is defined in `configs/a2.yaml` (model, epochs, learning rate, augmentation, device). Seeds 42-46 are the default.

```bash
python train.py --smoke                     # 5 batches per model, written to outputs_smoke/ (never evaluated)
# main study, CPU (14 threads): Experiments 1 and 2
python train.py CNN ResNet18Frozen ResNet18FineTune ResNet18FineTuneStrongAug
# follow-up study, one GPU per process: hardware check and Experiments 3-5
python train.py G_CNN G_ResNet18Frozen G_ResNet18FineTune G_ResNet18FrozenLR1e-4 G_ResNet18FineTuneLR1e-3 \
    G_ResNet18FineTune8ep G_ResNet18FineTune8epStrongAug G_EfficientNetB0FineTune G_ConvNeXtTinyFineTune
python train.py G_ConvNeXtTinyFineTune --seeds 42   # one run and one seed
python evaluate.py                          # validation + test metrics, predictions and inference time for every new run
jupyter nbconvert --to notebook --execute --inplace 02_results.ipynb 03_analysis.ipynb
```

Each run writes `Assignment2/outputs/<run>_seed<seed>_<timestamp>/` with `best.pt` (selected weights), `checkpoint_info.json` (configuration, git commit, environment), `history.csv`, `learning_curves.png`, `metrics.json` and the validation / test predictions. The console output of the 20 CPU runs is in `Assignment2/logs/`.

| Notebook | Content |
|---|---|
| `01_eda.ipynb` | EDA: class balance, image sizes and sources, colour, duplicates and leakage, preprocessing |
| `02_results.ipynb` | Main study (CPU, 20 runs): model comparison, augmentation ablation, leakage check, error taxonomy, compute cost |
| `03_analysis.ipynb` | Follow-up study (GPU, 45 runs): hardware check, frozen vs full at a matched schedule, augmentation with 8 epochs, EfficientNet-B0 and ConvNeXt-T |
| `02_results_m2.ipynb` | Archived M2 draft results (one seed) |

### Recorded environment

| | Main study (E1, E2) | Follow-up study (E3-E5) |
|---|---|---|
| Hardware | one CPU, 14 PyTorch threads | NVIDIA A40 48 GB (one run per card), AMD EPYC 7543 host |
| Source | commit `6869700` (tag `a2-final-runs`) | recorded in each `checkpoint_info.json` |
| Software | recorded in each `checkpoint_info.json` | Python 3.12.16, PyTorch 2.14.0 (CUDA 13.0), torchvision 0.29.2, NumPy 2.5.3, scikit-learn 1.9.1 |
| Precision | float32 | float32, cuDNN deterministic mode, no mixed precision |

Checkpoint selection: highest validation macro-F1, earliest epoch on a tie. The test set is evaluated once per run after selection.

### Checkpoints

The `outputs/` folders are not tracked by git (about 45 MB per ResNet-18 run, 111 MB per ConvNeXt-T run). The checkpoints are rebuilt exactly from the configuration and seed with the commands above; every run folder then holds its `best.pt`.

## Assignment 3: knowledge-based visual question answering

Reports: `docs/reports/A3_M1.pdf` (proposal), `A3_M2.pdf` (draft), `A3_M3.pdf` (final) · Slides: `docs/reports/A3_M3_slides.pdf` · Page: https://quin210.github.io/dl-co3133/assignment3.html

Dataset: A-OKVQA (`HuggingFaceM4/A-OKVQA`, revision `d1b0efa`): COCO images with multiple-choice questions that need commonsense or world knowledge. The official test split has no labels, so the official validation split is our test set and the official train split is divided into train and validation by image (seed 42).

```bash
cd Assignment3
python download.py      # parquet files -> data/aokvqa/
python data_loader.py   
python features.py      
python train.py         
python train.py LateFusion --seeds 42   # one run
jupyter nbconvert --to notebook --execute --inplace 01_eda.ipynb 02_results.ipynb
```

Each run folder holds `best.pt` (selected head weights), `metrics.json` (settings, commit, environment, validation and test results), `history.csv` and the test scores. Console logs are in `Assignment3/logs/`. Checkpoints are rebuilt exactly with the commands above. Hardware for the reported runs: one NVIDIA A40 per process.
