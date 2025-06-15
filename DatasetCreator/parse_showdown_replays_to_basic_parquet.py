import os
import re
import pandas as pd
from bs4 import BeautifulSoup
from utils.game_parser import extract_battle_log_from_html, parse_replay_data, extract_features, refine_features, generate_parquet_rows


REPLAY_DIR = "../Replays"
OUTPUT_PARQUET = "parsed_showdown_replays.parquet"
REVEAL_LIMIT = 1  # Set how many revealed Pokémon to include per team
IGNORE_LEAD = True  # If True, exclude the first revealed Pokémon (the lead)
TURN_LIMIT = -2  # -1 = all turns, -2 = until REVEAL_LIMIT is met, >0 = fixed number of turns
REVEAL_MODE = "p1"  # one of: "both", "p1", "p2", "either"

def main():
    all_rows = []
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
        except Exception as e:
            print(f"Failed to process {fname}: {e}")

    df = pd.DataFrame(all_rows)
    df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"Saved {len(df)} rows to {OUTPUT_PARQUET}")

if __name__ == "__main__":
    main()
