import os
import re
from typing import List, TypedDict

import pandas as pd
import copy
import numpy as np
from bs4 import BeautifulSoup

# Given a parquet where each entry is a turn, create a new dataset that focuses on what information gets revealed

INPUT_PARQUET = "./data/processed/Parquets/all_pokemon_showdown_replays.parquet"
OUTPUT_CSV = "./data/processed/Parquets/all_pokemon_moves.csv"

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


def main():
    print(f"Loading data from {INPUT_PARQUET}...")
    df = pd.read_parquet(INPUT_PARQUET)

    gameIds = df['game_id'].unique()
    print(f"Found {len(gameIds)} unique games")

    games: List[GameData] = []

    # Process each game
    for gameId in gameIds:
        turns = df[df['game_id'] == gameId]
        game = GameData(turns)
        # Only add games with at least one P2 Pokémon revealed
        if game.getTotalP2PokemonRevealed() >= 1:
            games.append(game)

        # Debug log to show progress
        if len(games) % 50 == 0:
            print(f"Processed {len(games)} games so far...")

    print(f"Processed {len(games)} valid games")
    print("Creating final dataset...")

    data_frames = []
    for game in games:
        try:
            game_df = game.createDataFrame()
            data_frames.append(game_df)
        except Exception as e:
            print(f"Error processing game {game.game_id}: {e}")
            continue

    if data_frames:
        new_df = pd.concat(data_frames, axis=0)

        # Verify game_id is in the output
        if 'game_id' not in new_df.columns:
            print("WARNING: game_id column is missing from the final DataFrame!")
        else:
            print(f"Successfully included game_id column with {new_df['game_id'].nunique()} unique values")

        print(f"Saving {len(new_df)} rows to {OUTPUT_CSV}")
        new_df.to_csv(OUTPUT_CSV, index=False)
        print(f"Data saved to {OUTPUT_CSV}")
    else:
        print("No valid data frames to concatenate!")


if __name__ == "__main__":
    main()