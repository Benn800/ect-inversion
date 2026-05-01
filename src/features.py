from pathlib import Path
import re


def get_project_root() -> Path:
	return Path(__file__).resolve().parents[1]


def ensure_project_in_path() -> None:
	import sys
	root = str(get_project_root())
	if root not in sys.path:
		sys.path.insert(0, root)


ensure_project_in_path()

from src.data import load_data


def load_df():
	return load_data()


def extract_unique_freqs(df):
    unique_freq = []

    for col in df.columns[2:]:
        freq = int(col.split('_')[-1])

        if freq not in unique_freq:
            unique_freq.append(freq)

    return unique_freq


if __name__ == "__main__":
	df = load_df()
	print(extract_unique_freqs(df))