"""Lung nodule detection in CT scans with YOLO.

The pipeline has three steps, each runnable as a module:

1. ``python -m lung_nodule_detection.preprocess``: CT volumes to a YOLO dataset.
2. ``python -m lung_nodule_detection.augment``: balance the training split.
3. ``python -m lung_nodule_detection.train``: train a YOLO model on the dataset.
"""
