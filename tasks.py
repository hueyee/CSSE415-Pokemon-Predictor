from celery import Celery
import pandas as pd
from train_pokemon_models import preprocess_data, train_model, evaluate_model, save_model_package

app = Celery('tasks', broker='redis://localhost:6379/0')  # Adjust if needed

@app.task
def train_by_game_id(game_id, df_path, pokemon_idx, output_dir):
    print(f"Training on game_id {game_id} for Pokemon {pokemon_idx}")
    df = pd.read_csv(df_path)
    df = df[df['game_id'] == game_id]

    if df.empty:
        print(f"No data found for game_id {game_id}. Skipping.")
        return

    X, y, cat_feats, num_feats, _ = preprocess_data(df, pokemon_idx)
    model_info = train_model(X, y, cat_feats, num_feats, pokemon_idx)
    evaluate_model(model_info, pokemon_idx)
    save_model_package(model_info, output_dir, pokemon_idx)

@app.task
def train_by_game_id_batch(game_id_batch, df_path, pokemon_idx, output_dir):
    print(f"Training batch of {len(game_id_batch)} games for Pokémon {pokemon_idx}")
    df = pd.read_csv(df_path)
    df = df[df['game_id'].isin(game_id_batch)]

    if df.empty:
        print("No data found for this batch. Skipping.")
        return

    X, y, cat_feats, num_feats, _ = preprocess_data(df, pokemon_idx)
    model_info = train_model(X, y, cat_feats, num_feats, pokemon_idx)
    evaluate_model(model_info, pokemon_idx)
    save_model_package(model_info, output_dir, pokemon_idx)
