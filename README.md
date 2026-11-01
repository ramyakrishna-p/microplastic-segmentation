# Microplastic Detection & Classification by Image Segmentation

**AI / Computer Vision | PyTorch, U-Net family — MSc Dissertation Project**

This is Ramya Krishna Parimi's real MSc dissertation project structure (Aston University, Jan–Apr 2026).

## Dissertation results (real, from her CV / dissertation)
- Dataset: **DeepParticle, 7,000+ annotated microplastic particles** from microscope images
- Models compared: **U-Net, U-Net++, DeepLabV3+, FPN** with ResNet-50 transfer learning, Dice-Focal Loss, data augmentation, class-imbalance handling, test-time augmentation
- Achieved: **41.8% mIoU and 53.3% Dice Score**

## Code in this repo
- `model.py` — PyTorch U-Net implementation (syntactically valid; install torch to instantiate)
- `train.py` — training skeleton / config explaining dataset setup (place DeepParticle images/masks under `data/images` and `data/masks`; the dataset is not included)
- `demo_metrics.py` — runnable IoU/Dice evaluation on **SYNTHETIC masks only**

## Demo results (SYNTHETIC, separate from dissertation)
Running `python demo_metrics.py` gives: demo mIoU **0.8336**, demo mean Dice **0.9089**, per-class IoU [0.8672, 0.8000], per-class Dice [0.9289, 0.8889]. These demo numbers are **not** dissertation results — they only verify the metric code runs.

## Run the demo
```bash
pip install -r requirements.txt
python demo_metrics.py
python train.py
```


## Original dissertation notebook
- `dissertation_plastic_segmentation.ipynb` - Ramya's full original MSc dissertation notebook (33 cells, with outputs): Plastic Detection and Classification by Image Segmentation on the DeepParticle microscope-image dataset, comparing U-Net, U-Net++, DeepLabV3+ and FPN with ResNet-50. Notebook test outputs include U-Net mIoU 0.4177 / mDice 0.5325, matching the dissertation results cited above (41.8% mIoU, 53.3% Dice).
- This project is MIT licensed (see LICENSE).
