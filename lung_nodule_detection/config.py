"""Constants shared by the preprocessing, augmentation and training steps."""

# CT intensity window applied before converting slices to 8-bit images.
WINDOW_CENTER_HU = -350
WINDOW_WIDTH_HU = 1000

# Nodule bounding boxes are enlarged around their centre, padded, and boxes
# closer than BOX_MERGE_DISTANCE_PX are merged into one.
BOX_SCALE = 1.3
BOX_PADDING_PX = 5
BOX_MERGE_DISTANCE_PX = 10

# Training slices whose nodules cover less than this area are dropped.
MIN_TRAIN_NODULE_AREA_MM2 = 25.0
# Slices used as augmentation sources must have at least this nodule area.
MIN_AUGMENT_NODULE_AREA_MM2 = 5.0

NODULE_CLASS_ID = 0
NODULE_CLASS_NAME = "nodule"

# Patient-wise train / validation / test split.
SPLIT_RATIOS = (0.7, 0.2, 0.1)
SPLITS = ("train", "val", "test")

LUNGMASK_MODEL = "R231"

# Files written by the preprocessing step next to the split folders.
DATASET_INFO_FILE = "dataset_info.json"
DATASET_YAML_FILE = "dataset.yaml"
