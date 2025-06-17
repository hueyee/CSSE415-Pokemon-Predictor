# tasks.py
from celery import Celery
import os
import pandas as pd
from train_pokemon_models import (
    split_data_with_validation, preprocess_data, train_model,
    evaluate_model, save_model_package
)

app = Celery('tasks', broker='redis://100.68.99.79:6379/0', backend='redis://100.68.99.79:6379/1')  # or your actual Redis URL

@app.task
def train_model_for_pokemon_idx(file_path, pokemon_idx, output_dir):
    print(f"Starting training for Pokemon {pokemon_idx}")
    df = pd.read_csv(file_path)
    df = split_data_with_validation(df)
    X, y, cat_feats, num_feats, _ = preprocess_data(df, pokemon_idx)
    model_info = train_model(X, y, cat_feats, num_feats, pokemon_idx)
    evaluate_model(model_info, pokemon_idx)
    save_model_package(model_info, output_dir, pokemon_idx)

