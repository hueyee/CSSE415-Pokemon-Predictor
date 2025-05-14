import os
import re
from typing import List, TypedDict

import pandas as pd
import copy
import numpy as np
from bs4 import BeautifulSoup

# Given a parquet where each entry is a turn, create a new dataset that focuses on what information gets revealed

INPUT_PARQUET = "../Parquets/all_pokemon_showdown_replays.parquet"

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
    'moves': List[str],
})

DataRevealDict = TypedDict('DataRevealDict', {
    'turn_id': int,
    'p1_current_pokemon': str,
    'p2_current_pokemon': str,
    'p1_pokemon': List[PokemonDict],
    'p2_pokemon': List[PokemonDict],
})


class GameData:

    def __init__(self, data: pd.DataFrame):

        self.turns: List[TurnDict] = []

        for index, turnData in data.iterrows():
            turn: TurnDict = turnData.to_dict()
            self.turns.append(turn)

        # This stores entries of how data gets revealed as the game progresses
        self.revealProgress: List[DataRevealDict] = []

        self.processTurns()

    def processTurns(self):
        # initialize current revealed info from turn 0
        currentRevealedInfo: DataRevealDict = {'turn_id': 0}

        p1_pokemon1: PokemonDict = {'name': self.turns[0]['p1_current_pokemon'], 'moves': []}
        p2_pokemon1: PokemonDict = {'name': self.turns[0]['p2_current_pokemon'], 'moves': []}

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
                new_pokemon: PokemonDict = {'name': turn['p1_current_pokemon'], 'moves': []}
                currentRevealedInfo['p1_pokemon'].append(new_pokemon)
                currentRevealedInfo['p1_current_pokemon'] = new_pokemon['name']
            else:
                currentRevealedInfo['p1_current_pokemon'] = turn['p1_current_pokemon']
                for pokemon in currentRevealedInfo['p1_pokemon']:
                    if pokemon['name'] == currentRevealedInfo['p1_current_pokemon']:
                        if turn['p1_move'] not in pokemon['moves']:
                            pokemon['moves'].append(turn['p1_move'])

            # Player 2 updates
            if len(currentRevealedInfo['p2_pokemon']) < turn['p2_number_of_pokemon_revealed']:
                # Assuming this is a new pokemon
                new_pokemon: PokemonDict = {'name': turn['p2_current_pokemon'], 'moves': []}
                currentRevealedInfo['p2_pokemon'].append(new_pokemon)
                currentRevealedInfo['p2_current_pokemon'] = new_pokemon['name']
            else:
                currentRevealedInfo['p2_current_pokemon'] = turn['p2_current_pokemon']
                for pokemon in currentRevealedInfo['p2_pokemon']:
                    if pokemon['name'] == currentRevealedInfo['p2_current_pokemon']:
                        if turn['p2_move'] not in pokemon['moves']:
                            pokemon['moves'].append(turn['p2_move'])

            currentRevealedInfo['turn_id'] = turn['turn_id']
            self.revealProgress.append(copy.deepcopy(currentRevealedInfo))

    def finalizeData(self):
        for revealData in self.revealProgress:

            for pokemon in revealData['p1_pokemon']:
                for i in range(4 - len(pokemon['moves'])):
                    pokemon['moves'].append(None)

            for i in range(6 - len(revealData['p1_pokemon'])):
                empty_pokemon: PokemonDict = {'name': None, 'moves': [None, None, None, None]}
                revealData['p1_pokemon'].append(empty_pokemon)

            for pokemon in revealData['p2_pokemon']:
                for i in range(4 - len(pokemon['moves'])):
                    pokemon['moves'].append(None)

            for i in range(6 - len(revealData['p2_pokemon'])):
                empty_pokemon: PokemonDict = {'name': None, 'moves': [None, None, None, None]}
                revealData['p2_pokemon'].append(empty_pokemon)
            pass

    def createDataFrame(self) -> pd.DataFrame:
        self.finalizeData()

        players = ['p1', 'p2']
        pokemonLabels = ['pokemon1', 'pokemon2', 'pokemon3', 'pokemon4', 'pokemon5', 'pokemon6']
        pokemonInfoLabels = ['name', 'move1', 'move2', 'move3', 'move4']

        p1_rating = self.turns[0]['p1_rating']
        p2_rating = self.turns[0]['p2_rating']

        # create the columns
        columns = ['turn_id']
        for player in players:
            columns.append(f'{player}_rating')
            columns.append(f'{player}_current_pokemon')

            for pokemon in pokemonLabels:

                for pokemonInfo in pokemonInfoLabels:
                    columns.append(f'{player}_{pokemon}_{pokemonInfo}')

        data = []
        # arrange data
        for revealData in self.revealProgress:
            dataEntry = [revealData['turn_id'], p1_rating, revealData['p1_current_pokemon']]

            for i in range(6):
                name = revealData['p1_pokemon'][i]['name']
                move1 = revealData['p1_pokemon'][i]['moves'][0]
                move2 = revealData['p1_pokemon'][i]['moves'][1]
                move3 = revealData['p1_pokemon'][i]['moves'][2]
                move4 = revealData['p1_pokemon'][i]['moves'][3]

                dataEntry.append(name)
                dataEntry.append(move1)
                dataEntry.append(move2)
                dataEntry.append(move3)
                dataEntry.append(move4)

            dataEntry.append(p2_rating)
            dataEntry.append(revealData['p2_current_pokemon'])

            for i in range(6):
                name = revealData['p2_pokemon'][i]['name']
                move1 = revealData['p2_pokemon'][i]['moves'][0]
                move2 = revealData['p2_pokemon'][i]['moves'][1]
                move3 = revealData['p2_pokemon'][i]['moves'][2]
                move4 = revealData['p2_pokemon'][i]['moves'][3]

                dataEntry.append(name)
                dataEntry.append(move1)
                dataEntry.append(move2)
                dataEntry.append(move3)
                dataEntry.append(move4)

            data.append(dataEntry)

        df = pd.DataFrame(data=data, columns=columns)

        return df


def main():
    df = pd.read_parquet(INPUT_PARQUET)

    gameIds = df['game_id'].unique()

    turns = df[df['game_id'] == gameIds[0]]

    gameData = GameData(turns)

    new_df = gameData.createDataFrame()
    new_df.to_csv('test.csv', index=False)

    # games: List[GameData] = []
    #
    # for gameId in gameIds:
    #     turns = df[df['game_id'] == gameId]
    #     games.append(GameData(turns))
    #
    # data = []
    # for game in games:
    #     data.append(game.createDataFrame())
    #
    # df = pd.concat(data, axis=0)
    #
    # df.to_csv('test.csv', index=False)


if __name__ == "__main__":
    main()
