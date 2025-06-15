import os
import re
import pandas as pd
from bs4 import BeautifulSoup
from utils.game_parser import extract_battle_log_from_html, parse_replay_data, extract_features, refine_features, generate_parquet_rows

REPLAY_DIR = "../Replays"
OUTPUT_PARQUET = "../Parquets/all_pokemon_showdown_replays.parquet"
TURN_LIMIT = -1  # -1 = all turns, >0 = fixed number of turns


def main():
    all_rows = []
    processed_files = 0

    for fname in os.listdir(REPLAY_DIR):
        if not fname.endswith(".html"):
            continue
        match = re.match(r"gen3ou-(.+?)\.html", fname)
        if not match:
            continue

        game_id = match.group(1)
        try:
            log = extract_battle_log_from_html(os.path.join(REPLAY_DIR, fname))
            turns = parse_replay_data(log)
            features = extract_features(turns)
            refined = refine_features(features)
            all_rows.extend(generate_parquet_rows(game_id, refined))
            processed_files += 1
            if processed_files % 10 == 0:
                print(f"Processed {processed_files} files...")
        except Exception as e:
            print(f"Failed to process {fname}: {e}")

    df = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(OUTPUT_PARQUET), exist_ok=True)
    df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"Saved {len(df)} rows to {OUTPUT_PARQUET}")

if __name__ == "__main__":
    main()