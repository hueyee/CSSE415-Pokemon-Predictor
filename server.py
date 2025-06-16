from flask import Flask, jsonify, request
import pandas as pd
from typing import List, TypedDict
import copy
import numpy as np
from bs4 import BeautifulSoup
import re
from prediction.predict import predict_team
from one_hot_encoder import CustomOneHotEncoder
from game_parser import parse_replay_data, extract_features, refine_features, generate_parquet_rows
from flask_cors import CORS

REVEAL_LIMIT = 1  # Set how many revealed Pokémon to include per team
IGNORE_LEAD = True  # If True, exclude the first revealed Pokémon (the lead)
TURN_LIMIT = -2  # -1 = all turns, -2 = until REVEAL_LIMIT is met, >0 = fixed number of turns
REVEAL_MODE = "p1"  # one of: "both", "p1", "p2", "either"



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
