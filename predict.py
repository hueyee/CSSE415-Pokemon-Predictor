import pandas as pd
import numpy as np
import joblib
import os
import glob
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

MODELS_DIR = "../../data/models/Models"
CONFIDENCE_THRESHOLD = 0.1


def load_latest_models():
    model_dirs = sorted(glob.glob(os.path.join(MODELS_DIR, "*")))
    if not model_dirs:
        raise ValueError(f"No model directories found in {MODELS_DIR}")
    latest_dir = model_dirs[-1]
    models = {}
    for pokemon_idx in range(2, 7):
        model_path = os.path.join(latest_dir, f"pokemon_prediction_model_{pokemon_idx}.joblib")
        if os.path.exists(model_path):
            models[pokemon_idx] = joblib.load(model_path)
        else:
            raise ValueError(f"Model for Pokemon {pokemon_idx} not found in {latest_dir}")
    return models

def get_relevant_features(row, model_info):
    features_dict = {}
    expected_features = model_info['categorical_features'] + model_info['numerical_features']
    for feature_name in expected_features:
        features_dict[feature_name] = [row.get(feature_name, np.nan)]
    return pd.DataFrame.from_dict(features_dict)

def predict_pokemon(row_features_df, model_info):
    encoder = model_info['encoder']
    categorical_features = model_info['categorical_features']
    numerical_features = model_info['numerical_features']

    X_cat_encoded = encoder.transform(row_features_df, categorical_features)

    X_num_values = np.zeros((row_features_df.shape[0], len(numerical_features)))
    for i, feature in enumerate(numerical_features):
        if feature in row_features_df.columns:
            X_num_values[:, i] = row_features_df[feature].fillna(0)

    X_combined = np.hstack([X_cat_encoded, X_num_values])
    model = model_info['model']
    proba = model.predict_proba(X_combined)[0]
    classes = model.classes_
    return {cls: prob for cls, prob in zip(classes, proba)}

def update_state(state, pokemon_idx, pokemon_name):
    new_state = state.copy()
    new_state[f'p2_pokemon{pokemon_idx}_name'] = pokemon_name
    new_state['p2_current_pokemon'] = pokemon_name
    new_state['p2_number_of_pokemon_revealed'] = pokemon_idx
    new_state['turn_id'] = new_state.get('turn_id', 0) + 3
    return new_state

def normalize_probabilities(prob_dict):
    if not prob_dict:
        return {}
    total_prob = sum(prob_dict.values())
    if total_prob == 0:
        return {k: 1/len(prob_dict) for k in prob_dict} if prob_dict else {}
    normalized = {k: v / total_prob for k, v in prob_dict.items()}
    return normalized

def create_prediction_visualization(prediction_summary, output_path="prediction_results.png"):
    if not prediction_summary or 'message' in prediction_summary:
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, prediction_summary.get('message', 'No predictions available'),
                 ha='center', va='center', fontsize=16)
        plt.title('Pokemon Team Prediction Results')
        plt.axis('off')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return

    slots = []
    pokemon_names = []
    confidences = []

    for slot_key, pred_data in prediction_summary.items():
        slot_num = slot_key.replace('slot_', '')
        slots.append(f'Slot {slot_num}')
        pokemon_names.append(pred_data['pokemon'])
        confidences.append(pred_data['confidence'])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    colors = plt.cm.viridis(np.linspace(0, 1, len(slots)))
    bars = ax1.bar(slots, confidences, color=colors, alpha=0.8, edgecolor='black', linewidth=1)

    ax1.set_title('Pokemon Prediction Confidence by Slot', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Pokemon Slot', fontsize=12)
    ax1.set_ylabel('Prediction Confidence', fontsize=12)
    ax1.set_ylim(0, max(confidences) * 1.2 if confidences else 1)
    ax1.grid(True, alpha=0.3)

    for bar, pokemon, conf in zip(bars, pokemon_names, confidences):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{pokemon}\n{conf:.3f}', ha='center', va='bottom',
                 fontweight='bold', fontsize=10)

    threshold_line = CONFIDENCE_THRESHOLD
    ax1.axhline(y=threshold_line, color='red', linestyle='--', alpha=0.7,
                label=f'Confidence Threshold ({threshold_line})')
    ax1.legend()

    slot_positions = range(len(slots))
    colors_confidence = ['green' if c >= 0.2 else 'orange' if c >= 0.1 else 'red' for c in confidences]

    ax2.scatter(slot_positions, confidences, c=colors_confidence, s=200, alpha=0.7, edgecolors='black', linewidth=2)

    for i, (pokemon, conf) in enumerate(zip(pokemon_names, confidences)):
        ax2.annotate(f'{pokemon[:10]}{"..." if len(pokemon) > 10 else ""}',
                     (i, conf), xytext=(5, 5), textcoords='offset points',
                     fontsize=9, ha='left')

    ax2.set_title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Pokemon Slot', fontsize=12)
    ax2.set_ylabel('Prediction Confidence', fontsize=12)
    ax2.set_xticks(slot_positions)
    ax2.set_xticklabels(slots)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=threshold_line, color='red', linestyle='--', alpha=0.7)

    legend_elements = [
        plt.scatter([], [], c='green', s=100, label='High Confidence (≥0.2)'),
        plt.scatter([], [], c='orange', s=100, label='Medium Confidence (≥0.1)'),
        plt.scatter([], [], c='red', s=100, label='Low Confidence (<0.1)')
    ]
    ax2.legend(handles=legend_elements, loc='upper right')

    plt.suptitle('Pokemon Team Prediction Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Prediction visualization saved to {output_path}")

def create_detailed_prediction_chart(results, all_paths, output_path="detailed_predictions.png"):
    if not results:
        return

    fig, axes = plt.subplots(len(results), 1, figsize=(15, 5*len(results)))
    if len(results) == 1:
        axes = [axes]

    for idx, result in enumerate(results):
        ax = axes[idx]
        slot = result['slot']
        predictions = result.get('predictions', [])

        if not predictions:
            ax.text(0.5, 0.5, f'No predictions for Slot {slot}',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Slot {slot} - No Predictions')
            continue

        top_predictions = predictions[:8]
        pokemon_names = []
        confidences = []

        for pred in top_predictions:
            if isinstance(pred, tuple) and len(pred) >= 2:
                pokemon_names.append(pred[0][:15] + ('...' if len(pred[0]) > 15 else ''))
                confidences.append(pred[1])
            else:
                pokemon_names.append(str(pred)[:15])
                confidences.append(0.0)

        if not confidences:
            ax.text(0.5, 0.5, f'No valid predictions for Slot {slot}',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Slot {slot} - No Valid Predictions')
            continue

        colors = plt.cm.plasma(np.linspace(0, 1, len(pokemon_names)))
        bars = ax.bar(pokemon_names, confidences, color=colors, alpha=0.8, edgecolor='black')

        ax.set_title(f'Slot {slot} - Top Predictions', fontsize=12, fontweight='bold')
        ax.set_ylabel('Confidence')
        ax.set_ylim(0, max(confidences) * 1.2 if confidences else 1)
        ax.grid(True, alpha=0.3)

        for bar, conf in zip(bars, confidences):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{conf:.3f}', ha='center', va='bottom', fontsize=9)

        ax.tick_params(axis='x', rotation=45)
        ax.axhline(y=CONFIDENCE_THRESHOLD, color='red', linestyle='--', alpha=0.5)

    plt.suptitle('Detailed Predictions by Slot', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Detailed prediction chart saved to {output_path}")

def predict_team(p1_rating, p2_rating, p2_pokemon1_name,
                 p2_pokemon2_name=None, p2_pokemon3_name=None, p2_pokemon4_name=None,
                 p2_pokemon5_name=None, turn_id=1):

    models = load_latest_models()

    pokemon_list = [p2_pokemon1_name, p2_pokemon2_name, p2_pokemon3_name,
                    p2_pokemon4_name, p2_pokemon5_name, None]

    revealed_count = 0
    for i, pokemon in enumerate(pokemon_list[:-1]):
        if pokemon is not None:
            revealed_count = i + 1
        else:
            break

    if revealed_count >= 6:
        prediction_summary = {"message": "All 6 Pokemon already provided - nothing to predict"}
        create_prediction_visualization(prediction_summary)
        return prediction_summary

    if revealed_count == 5:
        prediction_summary = {"message": "5 Pokemon provided - only 1 slot left, but models predict from slot 2 onwards"}
        create_prediction_visualization(prediction_summary)
        return prediction_summary

    initial_state = {
        'p1_rating': p1_rating,
        'p2_rating': p2_rating,
        'p2_current_pokemon': p2_pokemon1_name,
        'p2_number_of_pokemon_revealed': revealed_count,
        'turn_id': turn_id
    }

    for i in range(revealed_count):
        initial_state[f'p2_pokemon{i+1}_name'] = pokemon_list[i]

    start_idx = max(2, revealed_count + 1)

    results = []
    all_paths = {}

    for position_idx in range(start_idx, 7):
        model_to_use = models.get(position_idx)
        if not model_to_use:
            break

        if position_idx == start_idx:
            relevant_features_df = get_relevant_features(initial_state, model_to_use)
            direct_predictions = predict_pokemon(relevant_features_df, model_to_use)
            top_predictions = sorted(direct_predictions.items(), key=lambda x: x[1], reverse=True)
            candidate_predictions = [(p, pr) for p, pr in top_predictions if pr >= CONFIDENCE_THRESHOLD]
            if not candidate_predictions and top_predictions:
                candidate_predictions = [top_predictions[0]]

            current_result_predictions = []
            prediction_paths_for_next_step = []

            for pokemon, prob in candidate_predictions:
                new_state = update_state(initial_state, position_idx, pokemon)
                current_result_predictions.append((pokemon, prob))
                path = {'pokemon': pokemon, 'probability': prob, 'state': new_state, 'order': 0}
                prediction_paths_for_next_step.append(path)

            results.append({'slot': position_idx, 'predictions': current_result_predictions})
            all_paths[position_idx] = prediction_paths_for_next_step

        else:
            previous_slot_predictions = all_paths.get(position_idx - 1, [])
            if not previous_slot_predictions:
                break

            aggregated_probs_for_slot = defaultdict(float)
            parent_prob_sum_for_pokemon = defaultdict(float)

            for prev_path in previous_slot_predictions:
                prev_state = prev_path['state']
                prev_prob = prev_path['probability']

                relevant_prev_state_features_df = get_relevant_features(prev_state, model_to_use)
                new_raw_predictions = predict_pokemon(relevant_prev_state_features_df, model_to_use)

                top_new_raw_predictions = sorted(new_raw_predictions.items(), key=lambda x: x[1], reverse=True)
                candidate_new_predictions = [(p, pr) for p, pr in top_new_raw_predictions if pr >= CONFIDENCE_THRESHOLD]
                if not candidate_new_predictions and top_new_raw_predictions:
                    candidate_new_predictions = [top_new_raw_predictions[0]]

                for pokemon, current_step_prob in candidate_new_predictions:
                    new_state = update_state(prev_state, position_idx, pokemon)
                    aggregated_probs_for_slot[pokemon] += prev_prob * current_step_prob
                    parent_prob_sum_for_pokemon[pokemon] += prev_prob

            final_probs_for_slot = {}
            for pokemon, total_weighted_prob in aggregated_probs_for_slot.items():
                if parent_prob_sum_for_pokemon[pokemon] > 0:
                    final_probs_for_slot[pokemon] = total_weighted_prob / parent_prob_sum_for_pokemon[pokemon]

            normalized_final_probs = normalize_probabilities(final_probs_for_slot)
            top_normalized_predictions = sorted(normalized_final_probs.items(), key=lambda x: x[1], reverse=True)

            current_result_predictions = [(p, pr) for p, pr in top_normalized_predictions if pr >= CONFIDENCE_THRESHOLD]
            if not current_result_predictions and top_normalized_predictions:
                current_result_predictions = [top_normalized_predictions[0]]

            results.append({'slot': position_idx, 'predictions': current_result_predictions})

            prediction_paths_for_next_step = []
            for pokemon, normalized_prob in current_result_predictions:
                best_path = None
                highest_prob = -1
                for prev_path in previous_slot_predictions:
                    if aggregated_probs_for_slot[pokemon] > highest_prob:
                        best_path = prev_path
                        highest_prob = aggregated_probs_for_slot[pokemon]

                if best_path:
                    new_state = update_state(best_path['state'], position_idx, pokemon)
                    path = {'pokemon': pokemon, 'probability': normalized_prob, 'state': new_state}
                    prediction_paths_for_next_step.append(path)

            all_paths[position_idx] = prediction_paths_for_next_step
            if not prediction_paths_for_next_step:
                break

    prediction_summary = {}
    for result in results:
        slot = result['slot']
        if result['predictions']:
            top_pred = result['predictions'][0]
            prediction_summary[f'slot_{slot}'] = {
                'pokemon': top_pred[0],
                'confidence': round(top_pred[1], 4)
            }

    create_prediction_visualization(prediction_summary)
    create_detailed_prediction_chart(results, all_paths)

    return prediction_summary

if __name__ == "__main__":
    result = predict_team(
        p1_rating=1400,
        p2_rating=1600,
        p2_pokemon1_name="Pikachu",
        p2_pokemon2_name="Charizard",
        p2_pokemon3_name="Blastoise"
    )

    print("Prediction Result:")
    for slot, pred in result.items():
        if isinstance(pred, dict):
            print(f"  {slot}: {pred['pokemon']} (confidence: {pred['confidence']})")
        else:
            print(f"  {pred}")

    print("\nVisualizations created:")
    print("  - prediction_results.png")
    print("  - detailed_predictions.png")