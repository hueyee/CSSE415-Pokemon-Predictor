import pandas as pd
import os
from collections import Counter

FILE_PATH = "../Parquets/all_pokemon_moves.csv"

def count_pokemon_classes(file_path):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)

    # Check if the correct column exists
    target_column = "next_pokemon"
    if target_column not in df.columns:
        print(f"Error: Column '{target_column}' not found in the CSV file.")
        print(f"Available columns: {df.columns.tolist()}")
        return None

    # Get counts for the next_pokemon column
    unique_pokemons = df[target_column].dropna().unique()
    pokemon_count = len(unique_pokemons)
    top_pokemon = Counter(df[target_column].dropna()).most_common(20)

    # Print summary
    print("\n===== POKEMON CLASS COUNTS =====")
    print(f"Total unique Pokemon: {pokemon_count}")
    print(f"Top 20 most common Pokemon:")
    for pokemon, count in top_pokemon:
        print(f"  {pokemon}: {count} occurrences ({count/len(df.dropna(subset=[target_column]))*100:.2f}%)")

    # Analyze the distribution
    print("\n===== DISTRIBUTION ANALYSIS =====")
    total_non_null = df[target_column].notna().sum()
    print(f"Total non-null values: {total_non_null}")
    print(f"Missing values: {df[target_column].isna().sum()} ({df[target_column].isna().sum()/len(df)*100:.2f}%)")

    # Analyze by Pokemon revealed count
    if 'p2_number_of_pokemon_revealed' in df.columns:
        print("\n===== ANALYSIS BY POKEMON REVEALED =====")
        for revealed in sorted(df['p2_number_of_pokemon_revealed'].unique()):
            subset = df[df['p2_number_of_pokemon_revealed'] == revealed]
            if len(subset) > 0:
                non_null = subset[target_column].notna().sum()
                print(f"Pokemon revealed: {revealed}")
                print(f"  Rows: {len(subset)}")
                print(f"  Non-null next_pokemon: {non_null} ({non_null/len(subset)*100:.2f}%)")
                if non_null > 0:
                    print(f"  Unique Pokemon: {subset[target_column].dropna().nunique()}")
                    top5 = Counter(subset[target_column].dropna()).most_common(5)
                    print(f"  Top 5: {', '.join([f'{p[0]} ({p[1]})' for p in top5])}")

    return {
        'count': pokemon_count,
        'top_20': top_pokemon
    }

if __name__ == "__main__":
    if not os.path.exists(FILE_PATH):
        print(f"Error: File not found at {FILE_PATH}")
        print("Please update the FILE_PATH variable to point to your CSV file.")
    else:
        pokemon_counts = count_pokemon_classes(FILE_PATH)