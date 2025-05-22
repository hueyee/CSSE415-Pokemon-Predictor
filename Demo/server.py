# Import the Flask library
from flask import Flask, jsonify, request # Added request here for global access
import pandas as pd
from typing import List, TypedDict
import copy
import numpy as np
from bs4 import BeautifulSoup
import re
from Predict import predict_team
from flask_cors import CORS

REVEAL_LIMIT = 1  # Set how many revealed Pokémon to include per team
IGNORE_LEAD = True  # If True, exclude the first revealed Pokémon (the lead)
TURN_LIMIT = -2  # -1 = all turns, -2 = until REVEAL_LIMIT is met, >0 = fixed number of turns
REVEAL_MODE = "p1"  # one of: "both", "p1", "p2", "either"

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
            current_turn = [line]
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
                if len(parts) > 2 and parts[1] in ['switch', 'drag']:
                    player_pokemon = parts[2]
                    if ': ' in player_pokemon:
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
    }

    current_pokemon = {"p1": None, "p2": None}
    previous_pokemon = {
        "p1": [None, None, None, None, None],
        "p2": [None, None, None, None, None]
    }
    revealed_pokemon = {"p1": set(), "p2": set()}
    fainted_pokemon = {"p1": set(), "p2": set()}

    for feature in features:
        if "player_ratings" in feature:
            refined_features["player_ratings"] = feature["player_ratings"]
        else:
            turn_summary = {
                "active_pokemon": {"p1": current_pokemon["p1"], "p2": current_pokemon["p2"]},
                "previous_pokemon": {
                    "p1": previous_pokemon["p1"].copy(),
                    "p2": previous_pokemon["p2"].copy()
                },
                "num_revealed": {
                    "p1": len(revealed_pokemon["p1"]),
                    "p2": len(revealed_pokemon["p2"])
                },
                "moves_used": [],
                "damage": [],
                "statuses": [],
                "fainted": []
            }

            for event in feature["turn_events"]:
                parts = event.split('|')

                if len(parts) > 2 and parts[1] in ["switch", "drag"]:
                    player = parts[2].split(':')[0]
                    pokemon = parts[3].split(',')[0]

                    if "p1" in player:
                        if current_pokemon["p1"] != pokemon and current_pokemon["p1"] is not None:
                            previous_pokemon["p1"] = [current_pokemon["p1"]] + previous_pokemon["p1"][:-1]
                        current_pokemon["p1"] = pokemon
                        revealed_pokemon["p1"].add(pokemon)
                    elif "p2" in player:
                        if current_pokemon["p2"] != pokemon and current_pokemon["p2"] is not None:
                            previous_pokemon["p2"] = [current_pokemon["p2"]] + previous_pokemon["p2"][:-1]
                        current_pokemon["p2"] = pokemon
                        revealed_pokemon["p2"].add(pokemon)

                elif len(parts) > 2 and parts[1] == "move":
                    pokemon = parts[2].split(': ')[1] if ': ' in parts[2] else None
                    move = parts[3]
                    if pokemon:
                        turn_summary["moves_used"].append({"pokemon": pokemon, "move": move})

                elif len(parts) > 2 and parts[1] == "-damage":
                    target = parts[2].split(': ')[1] if ': ' in parts[2] else None
                    if target:
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
                    target = parts[2].split(': ')[1] if ': ' in parts[2] else None
                    status = parts[3]
                    if target:
                        turn_summary["statuses"].append({"target": target, "status": status})

                elif len(parts) > 2 and parts[1] == "faint":
                    pokemon_info = parts[2]
                    if ': ' in pokemon_info:
                        player, pokemon = pokemon_info.split(': ')[0], pokemon_info.split(': ')[1]
                        turn_summary["fainted"].append({"player": player, "pokemon": pokemon})
                        if "p1" in player:
                            fainted_pokemon["p1"].add(pokemon)
                        elif "p2" in player:
                            fainted_pokemon["p2"].add(pokemon)

            turn_summary["active_pokemon"] = current_pokemon.copy()
            turn_summary["previous_pokemon"] = {
                "p1": previous_pokemon["p1"].copy(),
                "p2": previous_pokemon["p2"].copy()
            }
            turn_summary["num_revealed"] = {
                "p1": len(revealed_pokemon["p1"]),
                "p2": len(revealed_pokemon["p2"])
            }
            turn_summary["fainted_pokemon"] = {
                "p1": list(fainted_pokemon["p1"]),
                "p2": list(fainted_pokemon["p2"])
            }

            refined_features["turns"].append(turn_summary)

    return refined_features

def generate_parquet_rows(game_id, refined):
    rows = []
    p1_rating = refined["player_ratings"].get("p1")
    p2_rating = refined["player_ratings"].get("p2")

    if TURN_LIMIT == -1:
        num_turns = len(refined["turns"])
    else:
        num_turns = min(TURN_LIMIT, len(refined["turns"]))

    for turn_id, turn in enumerate(refined["turns"][:num_turns]):
        p1_pokemon = turn["active_pokemon"]["p1"]
        p2_pokemon = turn["active_pokemon"]["p2"]

        p1_prev = turn["previous_pokemon"]["p1"]
        p2_prev = turn["previous_pokemon"]["p2"]

        p1_num_revealed = turn["num_revealed"]["p1"]
        p2_num_revealed = turn["num_revealed"]["p2"]

        p1_fainted = ",".join(turn["fainted_pokemon"]["p1"]) if turn["fainted_pokemon"]["p1"] else None
        p2_fainted = ",".join(turn["fainted_pokemon"]["p2"]) if turn["fainted_pokemon"]["p2"] else None

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

            "p1_current_pokemon": p1_pokemon,
            "p2_current_pokemon": p2_pokemon,

            "p1_first_previous_pokemon": p1_prev[0],
            "p1_second_previous_pokemon": p1_prev[1],
            "p1_third_previous_pokemon": p1_prev[2],
            "p1_fourth_previous_pokemon": p1_prev[3],
            "p1_fifth_previous_pokemon": p1_prev[4],

            "p2_first_previous_pokemon": p2_prev[0],
            "p2_second_previous_pokemon": p2_prev[1],
            "p2_third_previous_pokemon": p2_prev[2],
            "p2_fourth_previous_pokemon": p2_prev[3],
            "p2_fifth_previous_pokemon": p2_prev[4],

            "p1_number_of_pokemon_revealed": p1_num_revealed,
            "p2_number_of_pokemon_revealed": p2_num_revealed,

            "p1_fainted_pokemon": p1_fainted,
            "p2_fainted_pokemon": p2_fainted,

            "p1_move": p1_move,
            "p2_move": p2_move,
            "p1_damage_taken": p1_damage,
            "p1_status": p1_status,
            "p2_damage_taken": p2_damage,
            "p2_status": p2_status,
        })

    return rows

TurnDict = TypedDict('TurnDict', {
    'game_id': str,
    'turn_id': int,
    'p1_rating': int,
    'p2_rating': int,
    'p1_current_pokemon': str,
    'p2_current_pokemon': str,
    'p1_first_previous_pokemon': str,
    'p1_second_previous_pokemon': str,
    'p1_third_previous_pokemon': str,
    'p1_fourth_previous_pokemon': str,
    'p1_fifth_previous_pokemon': str,
    'p2_first_previous_pokemon': str,
    'p2_second_previous_pokemon': str,
    'p2_third_previous_pokemon': str,
    'p2_fourth_previous_pokemon': str,
    'p2_fifth_previous_pokemon': str,
    'p1_number_of_pokemon_revealed': int,
    'p2_number_of_pokemon_revealed': int,
    'p1_fainted_pokemon': List[str],
    'p2_fainted_pokemon': List[str],
    'p1_move': str,
    'p2_move': str,
    'p1_damage_taken': float,
    'p1_status': str,
    'p2_damage_taken': float,
    'p2_status': str,
})

PokemonDict = TypedDict('PokemonDict', {
    'name': str,
    'moves_revealed': int,
    'moves': List[str],
})

DataRevealDict = TypedDict('DataRevealDict', {
    'game_id': str,  # Added game_id to DataRevealDict
    'turn_id': int,
    'p1_current_pokemon': str,
    'p2_current_pokemon': str,
    'p1_pokemon': List[PokemonDict],
    'p2_pokemon': List[PokemonDict],
    'p1_number_of_pokemon_revealed': int,
    'p2_number_of_pokemon_revealed': int,
})


class GameData:

    def __init__(self, data: pd.DataFrame):
        self.game_id = None  # Initialize game_id
        self.turns: List[TurnDict] = []

        for index, turnData in data.iterrows():
            turn: TurnDict = turnData.to_dict()
            self.turns.append(turn)

            # Extract game_id from the first turn (should be the same for all turns in this game)
            if self.game_id is None and 'game_id' in turn:
                self.game_id = turn['game_id']

        # This stores entries of how data gets revealed as the game progresses
        self.revealProgress: List[DataRevealDict] = []

        self.processTurns()

    def isFullGame(self) -> bool:
        return self.revealProgress[-1]['p2_number_of_pokemon_revealed'] == 6

    def getTotalP2PokemonRevealed(self) -> int:
        return self.revealProgress[-1]['p2_number_of_pokemon_revealed']
    
    def processTurns(self):
        # initialize current revealed info from turn 0
        currentRevealedInfo: DataRevealDict = {
            'game_id': self.game_id,  # Include game_id in initial state
            'turn_id': 0,
            'p1_number_of_pokemon_revealed': 1,
            'p2_number_of_pokemon_revealed': 1
        }


        p1_pokemon1: PokemonDict = {'name': self.turns[0]['p1_current_pokemon'], 'moves_revealed': 0, 'moves': []}
        p2_pokemon1: PokemonDict = {'name': self.turns[0]['p2_current_pokemon'], 'moves_revealed': 0, 'moves': []}

        currentRevealedInfo['p1_current_pokemon'] = p1_pokemon1['name']
        currentRevealedInfo['p2_current_pokemon'] = p2_pokemon1['name']

        currentRevealedInfo['p1_pokemon'] = [p1_pokemon1]
        currentRevealedInfo['p2_pokemon'] = [p2_pokemon1]

        self.revealProgress.append((copy.deepcopy(currentRevealedInfo)))

        # iterate through rest of turns
        for turn in self.turns[1:]:

            # Player 1 updates
            if len(currentRevealedInfo['p1_pokemon']) < turn['p1_number_of_pokemon_revealed']:
                # Assuming this is a new pokemon
                new_pokemon: PokemonDict = {'name': turn['p1_current_pokemon'], 'moves_revealed': 0, 'moves': []}
                currentRevealedInfo['p1_pokemon'].append(new_pokemon)
                currentRevealedInfo['p1_current_pokemon'] = new_pokemon['name']
                currentRevealedInfo['p1_number_of_pokemon_revealed'] += 1
            else:
                currentRevealedInfo['p1_current_pokemon'] = turn['p1_current_pokemon']
                if turn['p1_move'] is None:
                    continue
                for pokemon in currentRevealedInfo['p1_pokemon']:
                    if pokemon['name'] == currentRevealedInfo['p1_current_pokemon']:
                        if turn['p1_move'] not in pokemon['moves']:
                            pokemon['moves'].append(turn['p1_move'])
                            pokemon['moves_revealed'] += 1

            # Player 2 updates
            if len(currentRevealedInfo['p2_pokemon']) < turn['p2_number_of_pokemon_revealed']:
                # Assuming this is a new pokemon
                new_pokemon: PokemonDict = {'name': turn['p2_current_pokemon'], 'moves_revealed': 0, 'moves': []}
                currentRevealedInfo['p2_pokemon'].append(new_pokemon)
                currentRevealedInfo['p2_current_pokemon'] = new_pokemon['name']
                currentRevealedInfo['p2_number_of_pokemon_revealed'] += 1
            else:
                currentRevealedInfo['p2_current_pokemon'] = turn['p2_current_pokemon']
                if turn['p2_move'] is None:
                    continue
                for pokemon in currentRevealedInfo['p2_pokemon']:
                    if pokemon['name'] == currentRevealedInfo['p2_current_pokemon']:
                        if turn['p2_move'] not in pokemon['moves']:
                            pokemon['moves'].append(turn['p2_move'])
                            pokemon['moves_revealed'] += 1

            currentRevealedInfo['turn_id'] = turn['turn_id']
            self.revealProgress.append(copy.deepcopy(currentRevealedInfo))

    def finalizeData(self):
        for revealData in self.revealProgress:

            for pokemon in revealData['p1_pokemon']:
                # pokemon['moves_revealed'] = len(pokemon['moves'])
                for i in range(4 - len(pokemon['moves'])):
                    pokemon['moves'].append(None)

            for i in range(6 - len(revealData['p1_pokemon'])):
                empty_pokemon: PokemonDict = {'name': None, 'moves_revealed': 0, 'moves': [None, None, None, None]}
                revealData['p1_pokemon'].append(empty_pokemon)

            for pokemon in revealData['p2_pokemon']:
                # pokemon['moves_revealed'] = len(pokemon['moves'])
                for i in range(4 - len(pokemon['moves'])):
                    pokemon['moves'].append(None)

            for i in range(6 - len(revealData['p2_pokemon'])):
                empty_pokemon: PokemonDict = {'name': None, 'moves_revealed': 0, 'moves': [None, None, None, None]}
                revealData['p2_pokemon'].append(empty_pokemon)
            pass

    def createDataFrame(self) -> pd.DataFrame:
        numberP2Pokemon = self.getTotalP2PokemonRevealed()

        self.finalizeData()

        players = ['p1', 'p2']
        pokemonLabels = ['pokemon1', 'pokemon2', 'pokemon3', 'pokemon4', 'pokemon5', 'pokemon6']
        pokemonInfoLabels = ['name', 'moves_revealed', 'move1', 'move2', 'move3', 'move4']

        p1_rating = self.turns[0]['p1_rating']
        p2_rating = self.turns[0]['p2_rating']

        # create the columns - add game_id as the first column
        columns = ['game_id', 'turn_id']
        for player in players:
            columns.append(f'{player}_rating')
            columns.append(f'{player}_current_pokemon')
            columns.append(f'{player}_number_of_pokemon_revealed')

            for pokemon in pokemonLabels:

                for pokemonInfo in pokemonInfoLabels:
                    columns.append(f'{player}_{pokemon}_{pokemonInfo}')

        data = []
        # arrange data
        for revealData in self.revealProgress:
            # if revealData['p2_number_of_pokemon_revealed'] >= numberP2Pokemon:
            #     break  # there are no more pokemon to predict so break

            pass
            # Include game_id as the first element
            dataEntry = [revealData['game_id'], revealData['turn_id'], p1_rating, revealData['p1_current_pokemon'],
                         revealData['p1_number_of_pokemon_revealed']]

            for i in range(6):
                name = revealData['p1_pokemon'][i]['name']
                moves_revealed = revealData['p1_pokemon'][i]['moves_revealed']
                move1 = revealData['p1_pokemon'][i]['moves'][0]
                move2 = revealData['p1_pokemon'][i]['moves'][1]
                move3 = revealData['p1_pokemon'][i]['moves'][2]
                move4 = revealData['p1_pokemon'][i]['moves'][3]

                dataEntry.append(name)
                dataEntry.append(moves_revealed)
                dataEntry.append(move1)
                dataEntry.append(move2)
                dataEntry.append(move3)
                dataEntry.append(move4)

            dataEntry.append(p2_rating)
            dataEntry.append(revealData['p2_current_pokemon'])
            dataEntry.append(revealData['p2_number_of_pokemon_revealed'])

            for i in range(6):
                name = revealData['p2_pokemon'][i]['name']
                moves_revealed = revealData['p2_pokemon'][i]['moves_revealed']
                move1 = revealData['p2_pokemon'][i]['moves'][0]
                move2 = revealData['p2_pokemon'][i]['moves'][1]
                move3 = revealData['p2_pokemon'][i]['moves'][2]
                move4 = revealData['p2_pokemon'][i]['moves'][3]

                dataEntry.append(name)
                dataEntry.append(moves_revealed)
                dataEntry.append(move1)
                dataEntry.append(move2)
                dataEntry.append(move3)
                dataEntry.append(move4)

            data.append(dataEntry)

        df = pd.DataFrame(data=data, columns=columns)

        # Go back through the data and create the "next_pokemon" column
        # NOTE: we only want to use turns where there is a next pokemon to reveal
        df['next_pokemon'] = None
        try:
            # Group by game_id first to ensure we're looking at the correct game sequence
            for game_id, game_group in df.groupby('game_id'):
                for i in range(game_group.iloc[-1]['p2_number_of_pokemon_revealed'] - 1):
                    # Get the first Pokémon that appears when p2_number_of_pokemon_revealed == i+2
                    next_pokemon = game_group[game_group['p2_number_of_pokemon_revealed'] == i + 2].iloc[0]['p2_current_pokemon'] if not game_group[game_group['p2_number_of_pokemon_revealed'] == i + 2].empty else None

                    # Set next_pokemon for all rows in this game where p2_number_of_pokemon_revealed == i+1
                    row_indices = df[(df['game_id'] == game_id) & (df['p2_number_of_pokemon_revealed'] == i + 1)].index
                    df.loc[row_indices, 'next_pokemon'] = next_pokemon
        except Exception as e:
            print(f"Error generating next_pokemon column: {e}")
            # Continue processing even if there's an error

        # Drop rows where next_pokemon is None
        df.dropna(subset=['next_pokemon'], inplace=True)

        return df

class CustomOneHotEncoder:
    def __init__(self):
        self.encoders = {}
        self.feature_names = []
        self.n_features = 0

    def fit(self, X, categorical_features):
        self.encoders = {}
        self.feature_names = []
        self.n_features = 0
        for feature in categorical_features:
            unique_values = X[feature].unique()
            self.encoders[feature] = {value: i for i, value in enumerate(unique_values)}
            for value in unique_values:
                self.feature_names.append(f"{feature}_{value}")
            self.n_features += len(unique_values)
        return self

    def transform(self, X, categorical_features):
        n_samples = X.shape[0]
        encoded = np.zeros((n_samples, self.n_features))
        current_idx = 0
        for feature in categorical_features:
            if feature not in self.encoders:
                current_idx += 0
                continue
            encoder = self.encoders[feature]
            for i, value in enumerate(X[feature]):
                if value in encoder:
                    encoded[i, current_idx + encoder[value]] = 1
            current_idx += len(encoder)
        return encoded

    def get_feature_names(self):
        return self.feature_names

# Create a Flask application instance
# __name__ is a special Python variable that gives the name of the current module.
# When you run your script directly, __name__ is set to "__main__".
# Flask uses this to know where to look for templates, static files, etc.
app = Flask(__name__)
CORS(app)

# Define a route that processes raw text from the request body
# This route expects a POST request to '/process-text'.
# The raw text data should be sent in the request body.
@app.route('/predict', methods=['POST'])
def process_text():
    """
    This function handles POST requests to the '/process-text' URL.
    It reads raw text data from the request body and returns a JSON response.
    """
    # Get the raw data from the request body
    raw_data = request.get_data()
    

    # Decode the raw data (assuming it's UTF-8 encoded text)
    # You might need to adjust the encoding based on what you expect.
    try:
        text_data = raw_data.decode('utf-8')
    except UnicodeDecodeError:
        return jsonify(error="Invalid UTF-8 data"), 400 # Bad request

    turns = parse_replay_data(text_data)
    features = extract_features(turns)
    refined = refine_features(features)
    turn_rows = generate_parquet_rows('current', refined)

    turn_df = pd.DataFrame(turn_rows)

    # print(turn_df.head())

    game = GameData(turn_df)

    df = game.createDataFrame()

    # print(df.head())

    last_turn = df.iloc[-1]

    p2_pokemon_revealed = last_turn['p2_number_of_pokemon_revealed']


    # for i in range(p2_pokemon_revealed):
    #     pokemon = last_turn[f'p2_pokemon{i+1}_name']
    #     print(pokemon)

    team = predict_team(
        p1_rating=last_turn['p1_rating'],
        p2_rating=last_turn['p2_rating'],
        p2_pokemon1_name=last_turn['p2_pokemon1_name'],
        p2_pokemon2_name=last_turn['p2_pokemon2_name'],
        p2_pokemon3_name=last_turn['p2_pokemon3_name'],
        p2_pokemon4_name=last_turn['p2_pokemon4_name'],
        p2_pokemon5_name=last_turn['p2_pokemon5_name'],
    )
    

    # You can now process the text_data as needed.
    # For this example, we'll just return the received text and its length.
    return jsonify(
        message="Text processed successfully",
        received_text=team,
        text_length=len(str(team))
    )

@app.route('/sample', methods=['GET'])
def sample_response():
    return jsonify(
        message="Text processed successfully",
        received_text=["Blissey", "Skarmory", "Tyranitar", "Gengar", "Swampert", "Jirachi"],
        text_length=len(str(["Blissey", "Skarmory", "Tyranitar", "Gengar", "Swampert", "Jirachi"]))
    )

# This block ensures that the Flask development server runs only
# when the script is executed directly (not when imported as a module).
if __name__ == '__main__':
    # app.run() starts the Flask development server.
    # debug=True enables debugging mode, which provides helpful error messages
    # and automatically reloads the server when code changes are detected.
    # host='0.0.0.0' makes the server accessible from any IP address,
    # which is useful if you want to access it from other devices on your network.
    # port=5000 specifies the port number the server will listen on.
    app.run(debug=True, host='0.0.0.0', port=5000)

# To run this API:
# 1. Make sure you have Flask installed: pip install Flask
# 2. Save this code as a Python file (e.g., api.py).
# 3. Open your terminal or command prompt.
# 4. Navigate to the directory where you saved the file.
# 5. Run the script using: python api.py
# 6. Open your web browser or a tool like Postman/curl.
#    - Test GET endpoints:
#      - http://127.0.0.1:5000/
#      - http://127.0.0.1:5000/greet/YourName
#      - http://127.0.0.1:5000/info?location=MyCity&topic=FlaskAPI
#    - Test POST endpoint (e.g., using curl):
#      curl -X POST -H "Content-Type: text/plain" --data "This is some raw text." http://127.0.0.1:5000/process-text
#      (This will send "This is some raw text." as the body to the /process-text endpoint)
