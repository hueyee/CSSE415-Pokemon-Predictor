from typing import List, TypedDict
import pandas as pd
import numpy as np

PokemonDict = TypedDict('PokemonDict', {'name': str, 'moves': List[str]})



PARQUET_FOLDER = "Parquets/"
FILE_PATH = PARQUET_FOLDER + "all_pokemon_showdown_replays.parquet"

df = pd.read_parquet(FILE_PATH)