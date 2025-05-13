import pandas as pd
import numpy as np

PARQUET_FOLDER = "Parquets/"
FILE_PATH = PARQUET_FOLDER + "all_pokemon_showdown_replays.parquet"

df = pd.read_parquet(FILE_PATH)

for i in range(6):
  current_revealed_pokemon = i + 1

  selected = df[df['p2_number_of_pokemon_revealed'] == current_revealed_pokemon]

  parquet_name = f"{current_revealed_pokemon}_revealed.parquet"
  selected.to_parquet(PARQUET_FOLDER + parquet_name)
