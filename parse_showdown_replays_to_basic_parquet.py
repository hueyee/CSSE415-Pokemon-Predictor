import os
import re
import pandas as pd
from bs4 import BeautifulSoup

REPLAY_DIR = "Replays"
OUTPUT_PARQUET = "parsed_showdown_replays.parquet"
REVEAL_LIMIT = 1  # Set how many revealed Pokémon to include per team
IGNORE_LEAD = True  # If True, exclude the first revealed Pokémon (the lead)
TURN_LIMIT = -2  # -1 = all turns, -2 = until REVEAL_LIMIT is met, >0 = fixed number of turns
REVEAL_MODE = "p1"  # one of: "both", "p1", "p2", "either"

def extract_battle_log_from_html(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    soup = BeautifulSoup(html_content, 'html.parser')
    battle_log_data = soup.find('script', {'class': 'battle-log-data'})
    if not battle_log_data:
        raise ValueError("Battle log data not found in the HTML file.")
    return battle_log_data.text.strip()

def parse_replay_data(battle_log):
    turns = []
    player_ratings = {}
    current_turn = []
    in_battle = False

    for line in battle_log.splitlines():
        line = line.strip()
        if not in_battle:
            if line.startswith("|player|"):
                parts = line.split("|")
                if len(parts) > 4:
                    player_id = parts[2]
                    rating = parts[-1]
                    if rating.isdigit():
                        player_ratings[player_id] = int(rating)
            elif line.startswith("|start"):
                in_battle = True
        elif line.startswith("|turn|"):
            if current_turn:
                turns.append(current_turn)
            current_turn = []
        elif line:
            current_turn.append(line)

    if current_turn:
        turns.append(current_turn)

    turns.insert(0, {"player_ratings": player_ratings})
    return turns

def extract_features(turns):
    nickname_mapping = {}
    features = []

    for i, turn in enumerate(turns):
        if i == 0:
            features.append({"player_ratings": turn.get("player_ratings", {})})
        else:
            current_features = {"turn_events": []}
            for event in turn:
                if i == len(turns) - 1 and (
                    event.startswith("|raw|") or event.startswith("|l|") or event.startswith("|player|")
                ):
                    continue

                parts = event.split('|')
                if len(parts) > 2 and parts[1] == 'switch':
                    player_pokemon = parts[2]
                    nickname = player_pokemon.split(': ')[1]
                    canonical_name = parts[3].split(',')[0]
                    nickname_mapping[nickname] = canonical_name

                updated_event = event
                for nickname, canonical_name in nickname_mapping.items():
                    updated_event = updated_event.replace(nickname, canonical_name)
                current_features["turn_events"].append(updated_event)
            features.append(current_features)
    return features

def refine_features(features):
    refined_features = {
        "player_ratings": None,
        "turns": [],
        "revealed_pokemon": {"p1": [], "p2": []}
    }

    current_pokemon = {"p1": None, "p2": None}
    for feature in features:
        if "player_ratings" in feature:
            refined_features["player_ratings"] = feature["player_ratings"]
        else:
            turn_summary = {
                "active_pokemon": {"p1": current_pokemon["p1"], "p2": current_pokemon["p2"]},
                "moves_used": [],
                "damage": [],
                "statuses": []
            }

            for event in feature["turn_events"]:
                parts = event.split('|')

                if len(parts) > 2 and parts[1] == "switch":
                    player = parts[2].split(':')[0]
                    pokemon = parts[3].split(',')[0]
                    if "p1" in player:
                        current_pokemon["p1"] = pokemon
                        if pokemon not in refined_features["revealed_pokemon"]["p1"]:
                            refined_features["revealed_pokemon"]["p1"].append(pokemon)
                    elif "p2" in player:
                        current_pokemon["p2"] = pokemon
                        if pokemon not in refined_features["revealed_pokemon"]["p2"]:
                            refined_features["revealed_pokemon"]["p2"].append(pokemon)

                elif len(parts) > 2 and parts[1] == "move":
                    pokemon = parts[2].split(': ')[1]
                    move = parts[3]
                    turn_summary["moves_used"].append({"pokemon": pokemon, "move": move})

                elif len(parts) > 2 and parts[1] == "-damage":
                    target = parts[2].split(': ')[1]
                    raw_hp = parts[3].split()[0]
                    if raw_hp == "0":
                        hp_val = 0.0
                    elif "/" in raw_hp:
                        try:
                            current, total = map(int, raw_hp.split("/")[0:2])
                            hp_val = round(current / total, 4)
                        except:
                            hp_val = None
                    else:
                        hp_val = None
                    turn_summary["damage"].append({"target": target, "hp": hp_val})

                elif len(parts) > 2 and parts[1] == "-status":
                    target = parts[2].split(': ')[1]
                    status = parts[3]
                    turn_summary["statuses"].append({"target": target, "status": status})

            turn_summary["active_pokemon"] = current_pokemon.copy()
            refined_features["turns"].append(turn_summary)

    return refined_features

def generate_parquet_rows(game_id, refined):
    rows = []
    p1_rating = refined["player_ratings"].get("p1")
    p2_rating = refined["player_ratings"].get("p2")

    revealed_raw_p1 = refined["revealed_pokemon"]["p1"]
    revealed_raw_p2 = refined["revealed_pokemon"]["p2"]
    start_idx = 1 if IGNORE_LEAD else 0
    revealed_p1 = revealed_raw_p1[start_idx:start_idx + REVEAL_LIMIT]
    revealed_p2 = revealed_raw_p2[start_idx:start_idx + REVEAL_LIMIT]

    if TURN_LIMIT == -1:
        num_turns = len(refined["turns"])
    elif TURN_LIMIT == -2:
        num_turns = 0
        seen_p1 = set()
        seen_p2 = set()
        for turn in refined["turns"]:
            active = turn["active_pokemon"]
            if active["p1"]:
                seen_p1.add(active["p1"])
            if active["p2"]:
                seen_p2.add(active["p2"])

            cutoff_p1 = list(seen_p1)[1:] if IGNORE_LEAD and len(seen_p1) > 0 else list(seen_p1)
            cutoff_p2 = list(seen_p2)[1:] if IGNORE_LEAD and len(seen_p2) > 0 else list(seen_p2)

            if REVEAL_MODE == "both" and len(cutoff_p1) >= REVEAL_LIMIT and len(cutoff_p2) >= REVEAL_LIMIT:
                break
            elif REVEAL_MODE == "p1" and len(cutoff_p1) >= REVEAL_LIMIT:
                break
            elif REVEAL_MODE == "p2" and len(cutoff_p2) >= REVEAL_LIMIT:
                break
            elif REVEAL_MODE == "either" and (len(cutoff_p1) >= REVEAL_LIMIT or len(cutoff_p2) >= REVEAL_LIMIT):
                break

            num_turns += 1
    else:
        num_turns = min(TURN_LIMIT, len(refined["turns"]))

    for turn_id, turn in enumerate(refined["turns"][:num_turns]):
        p1_pokemon = turn["active_pokemon"]["p1"]
        p2_pokemon = turn["active_pokemon"]["p2"]

        p1_move = next((m["move"] for m in turn["moves_used"] if m["pokemon"] == p1_pokemon), None)
        p2_move = next((m["move"] for m in turn["moves_used"] if m["pokemon"] == p2_pokemon), None)

        p1_damage = next((d["hp"] for d in turn["damage"] if d["target"] == p1_pokemon), None)
        p2_damage = next((d["hp"] for d in turn["damage"] if d["target"] == p2_pokemon), None)

        p1_status = next((s["status"] for s in turn["statuses"] if s["target"] == p1_pokemon), None)
        p2_status = next((s["status"] for s in turn["statuses"] if s["target"] == p2_pokemon), None)

        rows.append({
            "game_id": game_id,
            "turn_id": turn_id,
            "p1_rating": p1_rating,
            "p2_rating": p2_rating,
            "p1_pokemon": p1_pokemon,
            "p2_pokemon": p2_pokemon,
            "p1_move": p1_move,
            "p2_move": p2_move,
            "p1_damage_taken": p1_damage,
            "p1_status": p1_status,
            "p2_damage_taken": p2_damage,
            "p2_status": p2_status,
            "p1_revealed_pokemon": ",".join(revealed_p1),
            "p2_revealed_pokemon": ",".join(revealed_p2),
        })

    return rows

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
