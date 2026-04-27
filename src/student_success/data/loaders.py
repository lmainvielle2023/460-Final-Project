"""Load raw student performance data from disk."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


RAW_FILENAMES = {
    "por": "student-por.csv",
    "mat": "student-mat.csv",
}


def read_raw_student_file(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=";")


def load_student_performance(raw_dir: str | Path, subject: str = "por") -> pd.DataFrame:
    raw_dir = Path(raw_dir)

    if subject == "combined":
        frames: list[pd.DataFrame] = []
        for name, filename in RAW_FILENAMES.items():
            path = raw_dir / filename
            frame = read_raw_student_file(path)
            frame["subject"] = name
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    if subject not in RAW_FILENAMES:
        raise ValueError(
            f"Unsupported subject '{subject}'. Expected one of "
            f"{sorted(list(RAW_FILENAMES) + ['combined'])}."
        )

    return read_raw_student_file(raw_dir / RAW_FILENAMES[subject])


def validate_raw_layout(raw_dir: str | Path) -> dict[str, bool]:
    raw_dir = Path(raw_dir)
    return {
        filename: (raw_dir / filename).exists()
        for filename in RAW_FILENAMES.values()
    }
