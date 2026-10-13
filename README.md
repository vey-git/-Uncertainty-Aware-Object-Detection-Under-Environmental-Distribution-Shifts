# Uncertainty-Aware Object Detection Under Environmental Distribution Shifts

Weapon detection with YOLO, used to study how well a detector's uncertainty tracks its
real performance as conditions change (lighting, blur, noise, occlusion, different datasets).

## Workflow

```bash
# 1. Convert the downloaded datasets into one YOLO dataset (data/weapons) plus configs (data/*.yaml)
python prepare_datasets.py

# 2. Fine-tune YOLO on it. Try a small run first to check everything works.
python train.py --fraction 0.05 --epochs 3
python train.py

# 3. Run the live webcam detector with the trained weights
python main.py --weights runs/weapons/weights/best.pt
```

## Datasets

| Folder | Source | Used as |
|---|---|---|
| `dataset/` | Kaggle "Weapon and Knife Detection" (137k images) | train / val / test |
| `HARIS_Weapon_Detection_Dataset_v2/` | HARIS Weapon Detection v2 | train / val |
| `archive-3/` | Weapon-type dataset (9 weapon types) | train / val |
| `archive/` | People with guns (CVAT XML, 11 images) | test only |
| `archive-2/` | CCTV scenes | not used, already contained in HARIS |

All sources are mapped to two classes: `0 = gun`, `1 = knife`. Exact duplicate images are
removed so that nothing used for training appears in a validation or test split.
`data/weapons/summary.txt` lists what ended up where.
