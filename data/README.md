# Data Layout

Place the UCI Student Performance CSV files here:

- `data/raw/student-por.csv`
- `data/raw/student-mat.csv`

The scaffold assumes the original semicolon-delimited files from the UCI repository.

Recommended processing flow:

1. keep untouched source files in `data/raw/`
2. save cleaned or merged tables in `data/interim/`
3. save modeling-ready tables in `data/processed/`
