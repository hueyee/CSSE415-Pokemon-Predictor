import pandas as pd
import numpy as np
import joblib
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import tqdm
import logging
import warnings

logging.getLogger('joblib').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

FILE_PATH = "../Parquets/all_pokemon_moves.csv"
MODELS_DIR = "../Models"
CONFIDENCE_THRESHOLD = 0.1
MAX_PROPAGATIONS = 4
SAMPLE_SIZE = 1000

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
            encoder = self.encoders[feature]
            for i, value in enumerate(X[feature]):
                if value in encoder:
                    encoded[i, current_idx + encoder[value]] = 1
            current_idx += len(encoder)

        return encoded

    def get_feature_names(self):
        return self.feature_names

def load_latest_models():
    model_dirs = sorted(glob.glob(os.path.join(MODELS_DIR, "*")))
    if not model_dirs:
        raise ValueError(f"No model directories found in {MODELS_DIR}")

    latest_dir = model_dirs[-1]
    print(f"Loading models from directory: {latest_dir}")

    models = {}

    for pokemon_idx in tqdm.tqdm(range(2, 7), desc="Loading models"):
        model_path = os.path.join(latest_dir, f"pokemon_prediction_model_{pokemon_idx}.joblib")
        if os.path.exists(model_path):
            models[pokemon_idx] = joblib.load(model_path)
        else:
            raise ValueError(f"Model for Pokemon {pokemon_idx} not found in {latest_dir}")

    return models

def prepare_features(row, pokemon_idx, model_info):
    features = {}

    categorical_features = model_info['categorical_features']
    numerical_features = model_info['numerical_features']

    for feature in categorical_features + numerical_features:
        if feature in row:
            features[feature] = row[feature]

    return features

def encode_features(features, model_info):
    categorical_features = model_info['categorical_features']
    numerical_features = model_info['numerical_features']
    encoder = model_info['encoder']

    X_cat = np.zeros((1, encoder.n_features))

    current_idx = 0
    for feature in categorical_features:
        if feature in features:
            value = features[feature]
            if feature in encoder.encoders and value in encoder.encoders[feature]:
                X_cat[0, current_idx + encoder.encoders[feature][value]] = 1
        current_idx += len(encoder.encoders[feature])

    X_num = np.zeros((1, len(numerical_features)))
    for i, feature in enumerate(numerical_features):
        if feature in features:
            X_num[0, i] = features[feature]

    X_combined = np.hstack([X_cat, X_num])
    return X_combined

def predict_with_model(features, model_info):
    X_combined = encode_features(features, model_info)
    model = model_info['model']
    proba = model.predict_proba(X_combined)[0]
    classes = model.classes_

    result = {cls: prob for cls, prob in zip(classes, proba)}
    return result

def propagate_predictions(row, models, start_idx=2):
    original_row = row.copy()
    results = []

    for pokemon_idx in range(start_idx, 7):
        if pokemon_idx == start_idx:
            predictions = predict_next_pokemon(original_row, pokemon_idx, models[pokemon_idx])
            top_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)

            candidates = [(pokemon, prob) for pokemon, prob in top_predictions if prob >= CONFIDENCE_THRESHOLD]
            if not candidates:
                candidates = [top_predictions[0]]

            result = {
                'pokemon_idx': pokemon_idx,
                'predictions': candidates,
                'propagation_order': 0
            }
            results.append(result)

            row[f'p2_pokemon{pokemon_idx}_name'] = candidates[0][0]
        else:
            previous_result = results[-1]
            propagation_options = []

            for propagation_order, (prev_pokemon, prev_prob) in enumerate(previous_result['predictions']):
                if propagation_order >= MAX_PROPAGATIONS:
                    break

                propagation_row = row.copy()
                propagation_row[f'p2_pokemon{pokemon_idx-1}_name'] = prev_pokemon

                predictions = predict_next_pokemon(propagation_row, pokemon_idx, models[pokemon_idx])
                weighted_predictions = {pokemon: prob * prev_prob for pokemon, prob in predictions.items()}

                propagation_options.append({
                    'pokemon': prev_pokemon,
                    'confidence': prev_prob,
                    'predictions': weighted_predictions,
                    'propagation_order': propagation_order + 1
                })

            combined_predictions = defaultdict(float)
            for option in propagation_options:
                for pokemon, prob in option['predictions'].items():
                    combined_predictions[pokemon] += prob

            for pokemon in combined_predictions:
                combined_predictions[pokemon] /= sum(option['confidence'] for option in propagation_options)

            top_predictions = sorted(combined_predictions.items(), key=lambda x: x[1], reverse=True)
            candidates = [(pokemon, prob) for pokemon, prob in top_predictions if prob >= CONFIDENCE_THRESHOLD]
            if not candidates:
                candidates = [top_predictions[0]]

            result = {
                'pokemon_idx': pokemon_idx,
                'predictions': candidates,
                'propagation_order': max(option['propagation_order'] for option in propagation_options)
            }
            results.append(result)

            row[f'p2_pokemon{pokemon_idx}_name'] = candidates[0][0]

    return results

def predict_next_pokemon(row, pokemon_idx, model_info):
    features = prepare_features(row, pokemon_idx, model_info)
    predictions = predict_with_model(features, model_info)
    return predictions

def evaluate_predictions(df, models):
    results = {
        'overall': {
            'correct': 0,
            'total': 0,
            'accuracy': 0
        }
    }

    for pokemon_idx in range(2, 7):
        results[f'pokemon_{pokemon_idx}'] = {
            'correct': 0,
            'total': 0,
            'accuracy': 0,
            'by_propagation_order': defaultdict(lambda: {'correct': 0, 'total': 0, 'accuracy': 0})
        }

    all_predictions = []

    valid_rows = df[df['p2_number_of_pokemon_revealed'] < 6].copy()
    total_rows = len(valid_rows)

    print(f"Processing {total_rows} game states for prediction...")

    for i, row in tqdm.tqdm(valid_rows.iterrows(), total=total_rows, desc="Evaluating predictions"):
        revealed_count = row['p2_number_of_pokemon_revealed']
        start_idx = revealed_count + 1
        if start_idx < 2:
            start_idx = 2

        row_dict = row.to_dict()
        prediction_results = propagate_predictions(row_dict, models, start_idx)

        for result in prediction_results:
            pokemon_idx = result['pokemon_idx']
            propagation_order = result['propagation_order']
            top_prediction = result['predictions'][0][0]
            target = row[f'next_pokemon']

            is_correct = top_prediction == target

            results['overall']['total'] += 1
            results['overall']['correct'] += int(is_correct)

            results[f'pokemon_{pokemon_idx}']['total'] += 1
            results[f'pokemon_{pokemon_idx}']['correct'] += int(is_correct)

            results[f'pokemon_{pokemon_idx}']['by_propagation_order'][propagation_order]['total'] += 1
            results[f'pokemon_{pokemon_idx}']['by_propagation_order'][propagation_order]['correct'] += int(is_correct)

            all_predictions.append({
                'pokemon_idx': pokemon_idx,
                'propagation_order': propagation_order,
                'predicted': top_prediction,
                'actual': target,
                'correct': is_correct,
                'confidence': result['predictions'][0][1]
            })

    results['overall']['accuracy'] = results['overall']['correct'] / results['overall']['total'] if results['overall']['total'] > 0 else 0

    for pokemon_idx in range(2, 7):
        key = f'pokemon_{pokemon_idx}'
        results[key]['accuracy'] = results[key]['correct'] / results[key]['total'] if results[key]['total'] > 0 else 0

        for order in results[key]['by_propagation_order']:
            order_results = results[key]['by_propagation_order'][order]
            order_results['accuracy'] = order_results['correct'] / order_results['total'] if order_results['total'] > 0 else 0

    return results, all_predictions

def visualize_results(results, all_predictions):
    plt.figure(figsize=(12, 6))

    pokemon_accuracies = [results[f'pokemon_{idx}']['accuracy'] for idx in range(2, 7)]
    pokemon_labels = [f'Pokemon {idx}' for idx in range(2, 7)]

    plt.bar(pokemon_labels, pokemon_accuracies)
    plt.title('Prediction Accuracy by Pokemon Position')
    plt.xlabel('Pokemon Position')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    for i, acc in enumerate(pokemon_accuracies):
        plt.text(i, acc + 0.02, f'{acc:.2f}', ha='center')

    plt.tight_layout()
    plt.savefig('accuracy_by_pokemon_position.png')
    plt.close()

    propagation_order_df = pd.DataFrame(all_predictions)
    plt.figure(figsize=(14, 8))

    order_accuracy = propagation_order_df.groupby(['pokemon_idx', 'propagation_order'])['correct'].mean().reset_index()
    order_accuracy = order_accuracy.pivot(index='propagation_order', columns='pokemon_idx', values='correct')

    sns.heatmap(order_accuracy, annot=True, cmap='YlGnBu', fmt='.2f', cbar_kws={'label': 'Accuracy'})
    plt.title('Prediction Accuracy by Pokemon Position and Propagation Order')
    plt.xlabel('Pokemon Position')
    plt.ylabel('Propagation Order')

    plt.tight_layout()
    plt.savefig('accuracy_by_propagation_order.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    confidence_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    propagation_order_df['confidence_bin'] = pd.cut(propagation_order_df['confidence'], bins=confidence_bins)

    conf_accuracy = propagation_order_df.groupby('confidence_bin')['correct'].mean()
    conf_counts = propagation_order_df.groupby('confidence_bin').size()

    ax = conf_accuracy.plot(kind='bar', color='skyblue')
    plt.title('Accuracy by Confidence Level')
    plt.xlabel('Confidence Range')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)

    for i, acc in enumerate(conf_accuracy):
        plt.text(i, acc + 0.02, f'{acc:.2f}\n(n={conf_counts.iloc[i]})', ha='center')

    plt.tight_layout()
    plt.savefig('accuracy_by_confidence.png')
    plt.close()

    print("\nOverall Accuracy:", results['overall']['accuracy'])
    print("\nAccuracy by Pokemon Position:")
    for idx in range(2, 7):
        print(f"Pokemon {idx}: {results[f'pokemon_{idx}']['accuracy']:.4f}")

    print("\nAccuracy by Pokemon Position and Propagation Order:")
    for idx in range(2, 7):
        print(f"\nPokemon {idx}:")
        for order, order_results in sorted(results[f'pokemon_{idx}']['by_propagation_order'].items()):
            print(f"  Order {order}: {order_results['accuracy']:.4f} ({order_results['correct']}/{order_results['total']})")

def main():
    print("Loading data...")
    df = pd.read_csv(FILE_PATH)

    if SAMPLE_SIZE is not None:
        print(f"Sampling {SAMPLE_SIZE} rows from {len(df)} total rows")
        strat_col = 'p2_number_of_pokemon_revealed'
        sample_size_per_group = min(SAMPLE_SIZE // df[strat_col].nunique(),
                                    df.groupby(strat_col).size().min())

        sampled_df = df.groupby(strat_col).apply(
            lambda x: x.sample(min(len(x), sample_size_per_group), random_state=42)
        ).reset_index(drop=True)

        if len(sampled_df) < SAMPLE_SIZE and len(sampled_df) < len(df):
            remaining = SAMPLE_SIZE - len(sampled_df)
            excluded = df[~df.index.isin(sampled_df.index)]
            additional = excluded.sample(min(remaining, len(excluded)), random_state=42)
            sampled_df = pd.concat([sampled_df, additional])

        df = sampled_df
        print(f"Sampled data shape: {df.shape}")

    print("Loading models...")
    models = load_latest_models()

    print("Evaluating models with propagation...")
    results, all_predictions = evaluate_predictions(df, models)

    print("Visualizing results...")
    visualize_results(results, all_predictions)

    print("\nEvaluation complete. Check the output files for visualizations.")

if __name__ == "__main__":
    main()