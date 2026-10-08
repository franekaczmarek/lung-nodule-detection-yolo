"""Patient identification and patient-wise dataset splitting."""

import random
import re
from collections import defaultdict
from collections.abc import Iterable

from .config import SPLIT_RATIOS, SPLITS

_PATIENT_ID = re.compile(r"LIDC[_-]IDRI[_-](\d+)", re.IGNORECASE)


def patient_id(filename: str) -> str | None:
    """Return the LIDC-IDRI patient number in a filename, without leading zeros."""
    match = _PATIENT_ID.search(filename)
    return str(int(match.group(1))) if match else None


def split_by_patient(
    filenames: Iterable[str],
    ratios: tuple[float, float, float] = SPLIT_RATIOS,
    seed: int | None = None,
) -> dict[str, dict[str, list[str]]]:
    """Split files into train, val and test so that no patient spans two splits.

    Returns ``{split: {patient_id: [filenames]}}``.
    """
    if len(ratios) != len(SPLITS) or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"Expected three split ratios summing to 1, got {ratios}")

    files_by_patient: dict[str, list[str]] = defaultdict(list)
    for filename in sorted(filenames):
        pid = patient_id(filename)
        if pid is None:
            raise ValueError(f"No LIDC-IDRI patient id in filename {filename!r}")
        files_by_patient[pid].append(filename)

    patients = sorted(files_by_patient, key=int)
    random.Random(seed).shuffle(patients)

    n_train = round(len(patients) * ratios[0])
    n_val = min(round(len(patients) * ratios[1]), len(patients) - n_train)
    groups = {
        "train": patients[:n_train],
        "val": patients[n_train : n_train + n_val],
        "test": patients[n_train + n_val :],
    }
    return {
        split: {pid: files_by_patient[pid] for pid in sorted(pids, key=int)} for split, pids in groups.items()
    }
