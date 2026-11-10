import pandas as pd
from pathlib import Path

DATA_DIR = Path("../../data/stage_4_data/text_classification")
IN_FILE = DATA_DIR / "reviews_clean.csv"
df = pd.read_csv(IN_FILE)
print(df.shape)                       # expect roughly 50,000 rows
print(df.groupby(["split", "label"]).size())
print(df["text"].str.split().str.len().describe())   # token length stats
# print(df["text"].iloc[0])             # eyeball a sample
# print(df["text"].str.contains("br").sum())           # leftover "<br />" debris?
print(df["text"].str.split().str.len().quantile([0.9, 0.95, 0.99]))