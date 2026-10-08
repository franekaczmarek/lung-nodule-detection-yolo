# YOLO Pipelines for Lung Nodule Detection

A pipeline that turns chest CT scans from the [LIDC-IDRI](https://www.cancerimagingarchive.net/collection/lidc-idri/) database into a YOLO dataset and trains a detector that finds lung nodules on axial slices. It was built for my Master's thesis, where I compared YOLOv5 and YOLOv10 for nodule detection.

## How it works

1. **Preprocessing** (`lung_nodule_detection/preprocess.py`)
   - Splits patients 70 / 20 / 10 into train, validation and test, so no patient appears in two splits.
   - Segments the lungs on every slice with the [lungmask](https://github.com/JoHof/lungmask) R231 model and skips slices without both lungs.
   - Crops each slice to the lungs, applies a CT window (centre -350 HU, width 1000 HU) and converts it to 8-bit.
   - Resizes every crop onto one canvas, the size of the largest lung crop in the data, keeping the aspect ratio.
   - Writes a YOLO label per slice: nodule boxes are enlarged by 30%, padded by 5 px, and boxes closer than 10 px are merged. Slices without nodules get an empty label.
   - Leaves training slices with nodules smaller than 25 mm² out of the training set.
2. **Augmentation** (`lung_nodule_detection/augment.py`)
   - Most slices contain no nodule. For each training patient, rotated and zoomed copies of nodule slices are added until nodule and empty slices are balanced.
3. **Training** (`lung_nodule_detection/train.py`)
   - Trains an [Ultralytics](https://docs.ultralytics.com) YOLO model with the thesis hyperparameters (SGD, 100 epochs, batch 32, early stopping after 20 epochs).

## Project structure

```
lung_nodule_detection/
├── config.py       # windowing, box and split constants
├── patients.py     # patient ids and the patient-wise split
├── ct.py           # reading CT volumes, lung masks, windowing, resizing
├── boxes.py        # nodule boxes and YOLO label lines
├── dataset.py      # dataset folder layout and writing samples
├── preprocess.py   # step 1
├── augment.py      # step 2
└── train.py        # step 3
tests/              # unit tests and an end-to-end run on synthetic scans
```

## Usage

Requires Python 3.10+. Install the package with its dependencies:

```bash
pip install -e ".[dev]"
```

The input is two folders of NIfTI volumes: CT scans named like `LIDC_IDRI_0001_Images.nii.gz` and nodule masks with the same name but `Labels` instead of `Images`.

```bash
# 1. Build the dataset (writes train/, val/, test/ and dataset.yaml)
python -m lung_nodule_detection.preprocess \
    --images data/nifti/images --labels data/nifti/labels --output data/yolo

# 2. Balance the training split
python -m lung_nodule_detection.augment \
    --images data/nifti/images --labels data/nifti/labels --dataset data/yolo

# 3. Train YOLOv10-L (use yolov5lu.pt for YOLOv5-L)
python -m lung_nodule_detection.train --data data/yolo/dataset.yaml --weights yolov10l.pt
```

Every command has `--help`. The lungmask weights download automatically unless you pass `--lungmask-weights`.

Run the tests with `pytest`. They use synthetic scans and a stand-in lung segmenter, so they need neither the dataset nor a GPU.

## Results

- YOLOv10-L outperformed YOLOv5-L on mAP@0.5, precision, recall and inference time.
- The preprocessing (lung cropping, windowing and box enlargement) clearly improved detection.
- Most errors came from small nodules and nodules close to the lung boundary.

## Possible next steps

- Higher-resolution inputs or loss functions aimed at small objects.
- Validation on external datasets such as LUNA16 or NLST.
- Using 3D context from neighbouring slices.
- Benchmarks against Faster R-CNN or transformer-based detectors.
