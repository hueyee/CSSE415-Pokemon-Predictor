import os
import datetime
from tasks import train_model_for_pokemon_idx

FILE_PATH = "./data/processed/Parquets/all_pokemon_sequences.csv"
MODELS_DIR = "./data/models/Models"

def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(MODELS_DIR, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    for pokemon_idx in range(2, 7):
        train_model_for_pokemon_idx.delay(FILE_PATH, pokemon_idx, output_dir)

if __name__ == "__main__":
    main()
