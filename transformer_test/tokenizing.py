import pandas as pd
import json

df = pd.read_csv('Parquets/all_pokemon_moves.csv')

columns = []
players = ['p1', 'p2']
pokemonLabels = ['pokemon1', 'pokemon2', 'pokemon3', 'pokemon4', 'pokemon5', 'pokemon6']
pokemonInfoLabels = ['name', 'move1', 'move2', 'move3', 'move4']

# create the columns
for player in players:
    columns.append(f'{player}_current_pokemon')

    for pokemon in pokemonLabels:
        for pokemonInfo in pokemonInfoLabels:
            columns.append(f'{player}_{pokemon}_{pokemonInfo}')

vocab_df = df[columns]
vocabWords = vocab_df.stack()

tokenVocab = {}

currentInt = 1

for word in vocabWords.value_counts().keys():
    tokenVocab[word] = currentInt
    currentInt += 1

with open('tokens.json', 'w') as f:
    json.dump(tokenVocab, f)
