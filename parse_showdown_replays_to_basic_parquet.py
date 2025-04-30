import os
import pandas as pd
import re
from bs4 import BeautifulSoup

REPLAY_DIR = "Replays"
OUTPUT_FILE = "parsed_replays.parquet"

def extract_battle_log(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')
    log = soup.find('script', {'class': 'battle-log-data'})
    if not log:
        raise ValueError("Battle log not found")
    return log.text.strip()

def parse_turns(battle_log):
    lines = battle_log.splitlines()
    turns = []
    current_turn = []
    in_battle = False
    player_ratings = {"p1": None, "p2": None}

    for line in lines:
        line = line.strip()
        if not in_battle:
            if line.startswith("|player|"):
                parts = line.split("|")
                player = parts[2]
                rating = parts[-1]
                if rating.isdigit():
                    player_ratings[player] = int(rating)
            elif line.startswith("|start"):
                in_battle = True
        elif line.startswith("|turn|"):
            if current_turn:
                turns.append(current_turn)
            current_turn = []
        else:
            if line:
                current_turn.append(line)

    if current_turn:
        turns.append(current_turn)

    return player_ratings, turns[:5]

def parse_turn(turn_lines, nickname_to_pokemon):
    move_data = {"p1": None, "p2": None}
    damage_data = {"p1": None, "p2": None}
    status_data = {"p1": None, "p2": None}
    for line in turn_lines:
        parts = line.split("|")
        if len(parts) < 2:
            continue
        event = parts[1]

        if event == "switch":
            player = parts[2].split(":")[0]
            nickname = parts[2].split(": ")[1]
            pokemon = parts[3].split(",")[0]
            nickname_to_pokemon[nickname] = pokemon

        elif event == "move":
            player = parts[2].split(":")[0]
            nickname = parts[2].split(": ")[1]
            move = parts[3]
            mon = nickname_to_pokemon.get(nickname, nickname)
            move_data[player] = {"pokemon": mon, "move": move}

        elif event == "-damage":
            player_prefix = parts[2].split(":")[0]
            nickname = parts[2].split(": ")[1]
            damage = parts[3]
            if damage == "0 fnt":
                damage = "100/100"
            mon = nickname_to_pokemon.get(nickname, nickname)
            if "p1" in player_prefix:
                damage_data["p1"] = {"target": mon, "hp": damage}
            else:
                damage_data["p2"] = {"target": mon, "hp": damage}

        elif event == "-status":
            player_prefix = parts[2].split(":")[0]
            nickname = parts[2].split(": ")[1]
            status = parts[3]
            mon = nickname_to_pokemon.get(nickname, nickname)
            if "p1" in player_prefix:
                status_data["p1"] = {"target": mon, "status": status}
            else:
                status_data["p2"] = {"target": mon, "status": status}

    return move_data, damage_data, status_data

def get_second_revealed(turns, side):
    seen = []
    for turn in turns:
        for line in turn:
            if "|switch|" in line:
                parts = line.split("|")
                if len(parts) > 3:
                    player_prefix = parts[2].split(":")[0]
                    if side in player_prefix:
                        pokemon = parts[3].split(",")[0]
                        if pokemon not in seen:
                            seen.append(pokemon)
                            if len(seen) == 2:
                                return pokemon
    return None

def process_all_replays():
    rows = []
    for fname in os.listdir(REPLAY_DIR):
        if not fname.endswith(".html"):
            continue
        fpath = os.path.join(REPLAY_DIR, fname)
        match = re.match(r"gen3ou-(.+)\.html", fname)
        if not match:
            continue
        game_id = match.group(1)

        try:
            log = extract_battle_log(fpath)
            ratings, turns = parse_turns(log)
            p1_rating, p2_rating = ratings.get("p1"), ratings.get("p2")
            nickname_map = {}
            p1_revealed = get_second_revealed(turns, "p1")
            p2_revealed = get_second_revealed(turns, "p2")

            for turn_id, turn_lines in enumerate(turns):
                move, damage, status = parse_turn(turn_lines, nickname_map)

                row = {
                    "game_id": game_id,
                    "turn_id": turn_id,
                    "p1_rating": p1_rating,
                    "p2_rating": p2_rating,
                    "p1_pokemon": move["p1"]["pokemon"] if move["p1"] else None,
                    "p1_move": move["p1"]["move"] if move["p1"] else None,
                    "p1_damage_taken": damage["p1"]["hp"] if damage["p1"] else None,
                    "p1_status": status["p1"]["status"] if status["p1"] else None,
                    "p2_pokemon": move["p2"]["pokemon"] if move["p2"] else None,
                    "p2_move": move["p2"]["move"] if move["p2"] else None,
                    "p2_damage_taken": damage["p2"]["hp"] if damage["p2"] else None,
                    "p2_status": status["p2"]["status"] if status["p2"] else None,
                    "p1_revealed_pokemon": p1_revealed,
                    "p2_revealed_pokemon": p2_revealed,
                }
                rows.append(row)

        except Exception as e:
            print(f"Error processing {fname}: {e}")

    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = process_all_replays()
    df.to_parquet(OUTPUT_FILE, index=False)
    print(f"Saved {len(df)} rows to {OUTPUT_FILE}")
