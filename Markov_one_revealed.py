import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load the data
file_path = 'parsed_showdown_replays (1).parquet'
data = pd.read_parquet(file_path)

# Group by game and collect Player 1's moves in sequence
game_sequences = data.groupby('game_id')['p1_move'].apply(list)

# Split the sequences into training and testing sets
train_sequences, test_sequences = train_test_split(game_sequences, test_size=0.2, random_state=42)

# Build the Markov Chain transitions (training only)
transitions = defaultdict(lambda: defaultdict(int))

for sequence in train_sequences:
    for current_move, next_move in zip(sequence[:-1], sequence[1:]):
        if current_move is not None and next_move is not None:
            transitions[current_move][next_move] += 1

# Normalize the transitions to probabilities
markov_chain = {
    k: {k2: v2 / sum(v.values()) for k2, v2 in v.items()} 
    for k, v in transitions.items()
}

# === Accuracy Evaluation ===
correct_predictions = 0
total_predictions = 0

for sequence in test_sequences:
    for current_move, next_move in zip(sequence[:-1], sequence[1:]):
        if current_move in markov_chain:
            predicted_move = max(markov_chain[current_move], key=markov_chain[current_move].get)
            if predicted_move == next_move:
                correct_predictions += 1
            total_predictions += 1

accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
print(f"\n=== Markov Chain Accuracy ===\nTotal Predictions: {total_predictions}\nCorrect Predictions: {correct_predictions}\nAccuracy: {accuracy:.2%}")
