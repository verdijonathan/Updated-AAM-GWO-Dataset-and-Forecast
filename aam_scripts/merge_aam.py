import os
import pandas as pd


def normalize_and_load(csv_path: str) -> pd.DataFrame:
	"""Load a CSV and normalize to columns: date (YYYY-MM-DD-HH), aam."""
	df = pd.read_csv(csv_path)
	# Normalize column names if needed
	if "timestamp" in df.columns:
		df = df.rename(columns={"timestamp": "date"})
	if "aam_total_kg_m2_s" in df.columns:
		df = df.rename(columns={"aam_total_kg_m2_s": "aam"})
	# Ensure required columns
	missing = {"date", "aam"} - set(df.columns)
	if missing:
		raise ValueError(f"{csv_path} missing columns: {missing}")
	# Coerce and format datetime
	dt = pd.to_datetime(df["date"], errors="coerce")
	df = df.loc[dt.notna()].copy()
	df["date"] = dt.dt.strftime("%Y-%m-%d-%H")
	# Keep only required columns
	return df[["date", "aam"]]


def main() -> None:
	inputs = [
		"aam_1974_1982.csv",
		"aam_1983_1990.csv",
		"aam_climo.csv",          # 1991–2020 (your climo base, includes 1991)
		"aam_2021_2022.csv",
		"aam_2023.csv",
		"aam_2024.csv",
	]

	frames = []
	for path in inputs:
		if not os.path.exists(path):
			continue
		frames.append(normalize_and_load(path))

	if not frames:
		raise SystemExit("No input CSVs found to merge.")

	out = pd.concat(frames, ignore_index=True)
	# Drop bad rows and deduplicate by date; keep the first occurrence
	out = (
		out.dropna(subset=["date", "aam"])\
		.drop_duplicates(subset=["date"], keep="first")\
		.sort_values("date")
	)
	out.to_csv("aam_master_1974_2024.csv", index=False)
	print(f"wrote aam_master_1974_2024.csv with {len(out)} rows")


if __name__ == "__main__":
	main()


