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
import networkx as nx
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


logging.getLogger('joblib').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')


FILE_PATH = "../Parquets/all_pokemon_sequences.csv"  
MODELS_DIR = "../Models"
CONFIDENCE_THRESHOLD = 0.1
MAX_PROPAGATIONS = 4
SAMPLE_SIZE = 1000
DEBUG_MODE = True

class CustomOneHotEncoder:
    """Custom implementation of one-hot encoding for categorical features."""
    def __init__(self):
        self.encoders = {}
        self.feature_names = []
        self.n_features = 0

    def fit(self, X, categorical_features):
        """Fit the encoder to the data."""
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
        """Transform data using one-hot encoding."""
        n_samples = X.shape[0]
        encoded = np.zeros((n_samples, self.n_features))
        current_idx = 0
        for feature in categorical_features:
            if feature not in self.encoders: 
                current_idx += 0
                continue
            encoder = self.encoders[feature]
            for i, value in enumerate(X[feature]):
                if value in encoder:
                    encoded[i, current_idx + encoder[value]] = 1
            current_idx += len(encoder)
        return encoded

    def get_feature_names(self):
        """Get feature names after one-hot encoding."""
        return self.feature_names

def load_latest_models():
    """Load the most recent trained models from the models directory."""
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

def get_relevant_features(row, model_info):
    """Extract only the features needed by the model from the row data."""
    features_dict = {}
    expected_features = model_info['categorical_features'] + model_info['numerical_features']
    for feature_name in expected_features:
        features_dict[feature_name] = [row.get(feature_name, np.nan)]
    return pd.DataFrame.from_dict(features_dict)

def predict_pokemon(row_features_df, model_info):
    """Predict the next Pokemon using the model for the given position."""
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
    """Update the game state with a new Pokemon prediction."""
    new_state = state.copy()
    new_state[f'p2_pokemon{pokemon_idx}_name'] = pokemon_name
    new_state['p2_current_pokemon'] = pokemon_name
    new_state['p2_number_of_pokemon_revealed'] = pokemon_idx
    new_state['turn_id'] = new_state.get('turn_id', 0) + 3
    return new_state

def normalize_probabilities(prob_dict):
    """Normalize a dictionary of probabilities to make them sum to 1."""
    if not prob_dict:
        return {}
    total_prob = sum(prob_dict.values())
    if total_prob == 0:
        return {k: 1/len(prob_dict) for k in prob_dict} if prob_dict else {}
    normalized = {k: v / total_prob for k, v in prob_dict.items()}
    return normalized

def propagate_predictions(initial_state, models, start_idx=2):
    """
    Predict the full team composition starting from a partial team.
    Uses propagation to consider the impact of each prediction on subsequent ones.
    """
    results = []
    all_paths = {}
    viz_data = {'nodes': [], 'edges': [], 'game_states': []}
    root_id = 0
    starting_pokemon = initial_state.get(f'p2_pokemon{start_idx-1}_name') if start_idx > 1 else initial_state.get('p2_current_pokemon', 'Unknown')

    viz_data['nodes'].append({
        'id': root_id,
        'label': f"Start\n({initial_state.get('p2_number_of_pokemon_revealed',0)} rev)\n{starting_pokemon}",
        'type': 'start',
        'pokemon': starting_pokemon,
        'level': 0, 'order': 0
    })
    viz_data['game_states'].append({'node_id': root_id, 'state': initial_state.copy(), 'level': 0})
    current_node_id = 1

    if DEBUG_MODE:
        print(f"\n===== Starting Propagation from state with {initial_state.get('p2_number_of_pokemon_revealed', 0)} P2 Pokémon revealed =====")

    for position_idx in range(start_idx, 7):
        if DEBUG_MODE:
            print(f"\n--- Predicting Pokemon at effective slot {position_idx} ---")

        model_to_use = models.get(position_idx)
        if not model_to_use:
            if DEBUG_MODE: print(f"No model available for position {position_idx}. Stopping propagation.")
            break

        relevant_initial_features_df = get_relevant_features(initial_state, model_to_use)

        if position_idx == start_idx: 
            direct_predictions = predict_pokemon(relevant_initial_features_df, model_to_use)
            top_predictions = sorted(direct_predictions.items(), key=lambda x: x[1], reverse=True)
            candidate_predictions = [(p, pr) for p, pr in top_predictions if pr >= CONFIDENCE_THRESHOLD]
            if not candidate_predictions and top_predictions: candidate_predictions = [top_predictions[0]]

            current_result_predictions = []
            prediction_paths_for_next_step = []

            for pokemon, prob in candidate_predictions:
                new_state = update_state(initial_state, position_idx, pokemon)
                if DEBUG_MODE: print(f"  Direct prediction for slot {position_idx}: {pokemon} (conf: {prob:.4f})")

                current_result_predictions.append((pokemon, prob))

                path = {'pokemon': pokemon, 'probability': prob, 'state': new_state, 'node_id': current_node_id, 'order': 0}
                viz_data['nodes'].append({'id': current_node_id, 'label': f"{pokemon}\n{prob:.3f}", 'type': 'prediction', 'pokemon': pokemon, 'level': position_idx - start_idx +1, 'order': 0, 'probability': prob})
                viz_data['edges'].append({'source': root_id, 'target': current_node_id, 'type': 'prediction'})
                viz_data['game_states'].append({'node_id': current_node_id, 'state': new_state.copy(), 'level': position_idx - start_idx + 1})

                prediction_paths_for_next_step.append(path)
                current_node_id += 1

            results.append({'pokemon_idx': position_idx, 'predictions': current_result_predictions, 'propagation_order': 0})
            all_paths[position_idx] = prediction_paths_for_next_step

        else: 
            previous_slot_predictions = all_paths.get(position_idx - 1, [])
            if not previous_slot_predictions:
                if DEBUG_MODE: print(f"No previous paths to propagate from for slot {position_idx}. Stopping.")
                break

            current_slot_candidate_paths = []
            aggregated_probs_for_slot = defaultdict(float)
            parent_prob_sum_for_pokemon = defaultdict(float)

            for prev_path in previous_slot_predictions:
                prev_state = prev_path['state']
                prev_prob = prev_path['probability']
                prev_node_id = prev_path['node_id']
                prev_order = prev_path['order']

                relevant_prev_state_features_df = get_relevant_features(prev_state, model_to_use)
                new_raw_predictions = predict_pokemon(relevant_prev_state_features_df, model_to_use)

                top_new_raw_predictions = sorted(new_raw_predictions.items(), key=lambda x: x[1], reverse=True)
                candidate_new_predictions = [(p, pr) for p, pr in top_new_raw_predictions if pr >= CONFIDENCE_THRESHOLD]
                if not candidate_new_predictions and top_new_raw_predictions: candidate_new_predictions = [top_new_raw_predictions[0]]

                for pokemon, current_step_prob in candidate_new_predictions:
                    new_state = update_state(prev_state, position_idx, pokemon)

                    aggregated_probs_for_slot[pokemon] += prev_prob * current_step_prob
                    parent_prob_sum_for_pokemon[pokemon] += prev_prob

                    new_path = {
                        'pokemon': pokemon,
                        'current_step_prob': current_step_prob,
                        'parent_path_prob': prev_prob,
                        'state': new_state,
                        'node_id': current_node_id,
                        'parent_id': prev_node_id,
                        'order': prev_order + 1
                    }

                    viz_data['nodes'].append({'id': current_node_id, 'label': f"{pokemon}\n{current_step_prob:.3f}", 'type': 'prediction', 'pokemon': pokemon, 'level': position_idx - start_idx + 1, 'order': prev_order + 1, 'probability': current_step_prob})
                    viz_data['edges'].append({'source': prev_node_id, 'target': current_node_id, 'type': 'prediction'})
                    viz_data['game_states'].append({'node_id': current_node_id, 'state': new_state.copy(), 'level': position_idx - start_idx + 1})

                    current_slot_candidate_paths.append(new_path)
                    current_node_id += 1

            
            final_probs_for_slot = {}
            if not current_slot_candidate_paths:
                if DEBUG_MODE: print(f"No candidate paths generated for slot {position_idx}. Stopping.")
                break

            for pokemon, total_weighted_prob in aggregated_probs_for_slot.items():
                if parent_prob_sum_for_pokemon[pokemon] > 0:
                    final_probs_for_slot[pokemon] = total_weighted_prob / parent_prob_sum_for_pokemon[pokemon]

            normalized_final_probs = normalize_probabilities(final_probs_for_slot)
            top_normalized_predictions = sorted(normalized_final_probs.items(), key=lambda x: x[1], reverse=True)

            current_result_predictions = [(p, pr) for p, pr in top_normalized_predictions if pr >= CONFIDENCE_THRESHOLD]
            if not current_result_predictions and top_normalized_predictions: current_result_predictions = [top_normalized_predictions[0]]

            if DEBUG_MODE:
                print(f"  Propagated predictions for slot {position_idx} (top 3):")
                for p, pr in current_result_predictions[:3]: print(f"    {p}: {pr:.4f}")

            max_order_for_this_slot = max(p['order'] for p in current_slot_candidate_paths) if current_slot_candidate_paths else 0
            results.append({'pokemon_idx': position_idx, 'predictions': current_result_predictions, 'propagation_order': max_order_for_this_slot})

            
            prediction_paths_for_next_step = []
            avg_node_parent_map = defaultdict(list)

            for pokemon, normalized_prob in current_result_predictions:
                
                avg_node_id = current_node_id
                viz_data['nodes'].append({'id': avg_node_id, 'label': f"AVG {pokemon}\n{normalized_prob:.3f}", 'type': 'average', 'pokemon': pokemon, 'level': position_idx - start_idx + 1, 'probability': normalized_prob})
                current_node_id +=1

                best_path_for_this_pokemon = None
                highest_prob_sum = -1

                for path in current_slot_candidate_paths:
                    if path['pokemon'] == pokemon:
                        avg_node_parent_map[avg_node_id].append(path['node_id'])
                        current_path_strength = path['parent_path_prob'] * path['current_step_prob']
                        if best_path_for_this_pokemon is None or current_path_strength > highest_prob_sum:
                            best_path_for_this_pokemon = path
                            highest_prob_sum = current_path_strength

                if best_path_for_this_pokemon:
                    path_for_next = {
                        'pokemon': pokemon,
                        'probability': normalized_prob,
                        'state': best_path_for_this_pokemon['state'],
                        'node_id': avg_node_id,
                        'order': max_order_for_this_slot
                    }
                    prediction_paths_for_next_step.append(path_for_next)

            for avg_node_id, parent_nodes in avg_node_parent_map.items():
                for parent_node_id in parent_nodes:
                    viz_data['edges'].append({'source': parent_node_id, 'target': avg_node_id, 'type': 'average'})

            all_paths[position_idx] = prediction_paths_for_next_step
            if not prediction_paths_for_next_step:
                if DEBUG_MODE: print(f"No paths to continue propagation from slot {position_idx}. Stopping.")
                break

    if DEBUG_MODE: print("===== Propagation Finished =====")
    return results, viz_data

def create_prediction_visualization(viz_data, output_path="prediction_visualization.png"):
    """Create a network visualization of the prediction tree."""
    G = nx.DiGraph()
    node_colors, node_sizes, node_labels = [], [], {}
    for node in viz_data['nodes']:
        G.add_node(node['id'])
        node_labels[node['id']] = node['label']
        if node['type'] == 'start': node_colors.append('lightblue'); node_sizes.append(2000)
        elif node['type'] == 'average': node_colors.append('gold'); node_sizes.append(1800)
        elif node['type'] == 'prediction':
            colors = ['lightgreen', 'lightcoral', 'lightsalmon', 'khaki', 'plum']
            node_colors.append(colors[min(node['order'], len(colors)-1)])
            node_sizes.append(max(500, 1500 - node.get('order',0) * 200))
    for edge in viz_data['edges']: G.add_edge(edge['source'], edge['target'], style=edge.get('type', 'prediction'))

    pos = nx.multipartite_layout(G, subset_key="level")
    plt.figure(figsize=(min(30, G.number_of_nodes()*1.5), min(20, max(n['level'] for n in viz_data['nodes'])*2 + 5 if viz_data['nodes'] else 10 )))

    edge_styles = {"prediction": "solid", "average": "dashed"}
    for style, linestyle in edge_styles.items():
        edgelist = [(u,v) for u,v,data in G.edges(data=True) if data.get("style")==style]
        nx.draw_networkx_edges(G, pos, edgelist=edgelist, style=linestyle, width=1.0, alpha=0.7, arrows=True, arrowstyle='->', arrowsize=10)

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.9, edgecolors='gray')
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)

    legend_elements = [
        Patch(facecolor='lightblue', label='Start'), Patch(facecolor='lightgreen', label='Order 0'),
        Patch(facecolor='lightcoral', label='Order 1'), Patch(facecolor='lightsalmon', label='Order 2'),
        Patch(facecolor='khaki', label='Order 3+'), Patch(facecolor='gold', label='Weighted Avg'),
        Line2D([0], [0], color='black', lw=1, style='solid', label='Prediction'),
        Line2D([0], [0], color='black', lw=1, style='dashed', label='Aggregation')]
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.1, 1.0))
    plt.axis('off')
    plt.title("Pokemon Prediction Tree with Weighted Propagation")
    plt.tight_layout()
    try:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Prediction visualization saved to {output_path}")
    except Exception as e:
        print(f"Error saving visualization: {e}")
    plt.close()
    return G

def evaluate_predictions_with_sequences(df_full_dataset, models):
    """
    Evaluates Pokemon prediction accuracy for propagated predictions using the sequence data.
    Uses the new p2_next_reveal_N columns to properly evaluate each level of propagation.
    """
    results_summary = {
        'overall': {'correct': 0, 'total': 0, 'accuracy': 0.0}
    }
    for i in range(1, 7):
        results_summary[f'pokemon_{i}'] = {
            'correct': 0, 'total': 0, 'accuracy': 0.0,
            'by_propagation_order': defaultdict(lambda: {'correct': 0, 'total': 0, 'accuracy': 0.0})
        }

    detailed_predictions_log = []

    
    prediction_start_states = df_full_dataset[df_full_dataset['p2_number_of_pokemon_revealed'] < 6].copy()

    for _, initial_state_row in tqdm.tqdm(prediction_start_states.iterrows(),
                                          total=len(prediction_start_states),
                                          desc="Evaluating predictions"):

        initial_p2_revealed_count = initial_state_row['p2_number_of_pokemon_revealed']

        
        first_slot_to_predict_in_chain = initial_p2_revealed_count + 1

        
        model_start_prediction_idx = max(2, first_slot_to_predict_in_chain)

        if model_start_prediction_idx > 6:
            continue

        initial_state_dict = initial_state_row.to_dict()

        
        prediction_chain, _ = propagate_predictions(initial_state_dict, models, model_start_prediction_idx)

        
        for prediction_step_idx, prediction_step in enumerate(prediction_chain):
            slot_model_predicted = prediction_step['pokemon_idx']
            propagation_order = prediction_step['propagation_order']

            if not prediction_step['predictions']:
                continue

            top_model_prediction_name = prediction_step['predictions'][0][0]
            top_model_confidence = prediction_step['predictions'][0][1]

            
            
            
            sequence_col = f'p2_next_reveal_{propagation_order + 1}'

            if sequence_col in initial_state_row and not pd.isna(initial_state_row[sequence_col]):
                actual_pokemon_for_this_step = initial_state_row[sequence_col]

                is_correct = (top_model_prediction_name == actual_pokemon_for_this_step)

                
                results_summary['overall']['total'] += 1
                results_summary['overall']['correct'] += int(is_correct)

                
                slot_key = f'pokemon_{slot_model_predicted}'
                if slot_key in results_summary:
                    results_summary[slot_key]['total'] += 1
                    results_summary[slot_key]['correct'] += int(is_correct)

                    
                    order_stats = results_summary[slot_key]['by_propagation_order'][propagation_order]
                    order_stats['total'] += 1
                    order_stats['correct'] += int(is_correct)

                detailed_predictions_log.append({
                    'game_id': initial_state_row.get('game_id', 'Unknown'),
                    'initial_p2_revealed_count': initial_p2_revealed_count,
                    'predicted_slot_index': slot_model_predicted,
                    'propagation_order': propagation_order,
                    'predicted_pokemon': top_model_prediction_name,
                    'actual_pokemon': actual_pokemon_for_this_step,
                    'is_correct': is_correct,
                    'confidence': top_model_confidence,
                })

    
    if results_summary['overall']['total'] > 0:
        results_summary['overall']['accuracy'] = results_summary['overall']['correct'] / results_summary['overall']['total']

    for i in range(1, 7):
        slot_key = f'pokemon_{i}'
        if slot_key in results_summary:
            slot_data = results_summary[slot_key]
            if slot_data['total'] > 0:
                slot_data['accuracy'] = slot_data['correct'] / slot_data['total']

            for order_data in slot_data['by_propagation_order'].values():
                if order_data['total'] > 0:
                    order_data['accuracy'] = order_data['correct'] / order_data['total']

    return results_summary, detailed_predictions_log

def visualize_results(results, all_predictions_df):
    """Create visualizations of prediction results."""
    
    plt.figure(figsize=(12, 6))
    pokemon_accuracies = [results.get(f'pokemon_{idx}', {}).get('accuracy', 0) for idx in range(2, 7)]
    pokemon_labels = [f'P2 Slot {idx}' for idx in range(2, 7)]

    plt.bar(pokemon_labels, pokemon_accuracies, color='skyblue')
    plt.title('Prediction Accuracy by P2 Pokémon Slot (Models predict slots 2-6)')
    plt.xlabel('P2 Pokémon Slot Index')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.05)
    for i, acc in enumerate(pokemon_accuracies):
        plt.text(i, acc + 0.02, f'{acc:.3f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig('accuracy_by_pokemon_position.png')
    plt.close()

    if not all_predictions_df.empty:
        
        order_accuracy_df = pd.DataFrame(columns=['Slot', 'Order', 'Accuracy', 'Count'])

        for slot in range(2, 7):
            slot_key = f'pokemon_{slot}'
            if slot_key not in results:
                continue

            slot_data = results[slot_key]
            for order, order_data in slot_data['by_propagation_order'].items():
                if order_data['total'] > 0:
                    new_row = pd.DataFrame({
                        'Slot': [slot],
                        'Order': [order],
                        'Accuracy': [order_data['accuracy']],
                        'Count': [order_data['total']]
                    })
                    order_accuracy_df = pd.concat([order_accuracy_df, new_row], ignore_index=True)

        if not order_accuracy_df.empty:
            plt.figure(figsize=(14, 8))
            pivot_table = order_accuracy_df.pivot(index='Slot', columns='Order', values='Accuracy')
            sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', fmt='.3f', cbar_kws={'label': 'Accuracy'})
            plt.title('Prediction Accuracy by Slot and Propagation Order')
            plt.xlabel('Propagation Order')
            plt.ylabel('P2 Pokémon Slot')
            plt.tight_layout()
            plt.savefig('accuracy_by_slot_and_order.png')
            plt.close()

            
            plt.figure(figsize=(14, 8))
            count_pivot = order_accuracy_df.pivot(index='Slot', columns='Order', values='Count')
            sns.heatmap(count_pivot, annot=True, cmap='Greens', fmt='g', cbar_kws={'label': 'Number of Predictions'})
            plt.title('Number of Predictions by Slot and Propagation Order')
            plt.xlabel('Propagation Order')
            plt.ylabel('P2 Pokémon Slot')
            plt.tight_layout()
            plt.savefig('counts_by_slot_and_order.png')
            plt.close()

        
        order_accuracy = all_predictions_df.groupby('propagation_order')['is_correct'].agg(['mean', 'count']).reset_index()

        if not order_accuracy.empty:
            plt.figure(figsize=(10, 6))
            bar_plot = sns.barplot(x='propagation_order', y='mean', data=order_accuracy, color='skyblue')
            plt.title('Prediction Accuracy by Propagation Order (Across All Slots)')
            plt.xlabel('Propagation Order')
            plt.ylabel('Accuracy')
            plt.ylim(0, 1.05)

            for i, row in order_accuracy.iterrows():
                plt.text(i, row['mean'] + 0.02, f"{row['mean']:.3f}\n(n={row['count']})",
                         ha='center', va='bottom')

            plt.tight_layout()
            plt.savefig('accuracy_by_propagation_order.png')
            plt.close()

        
        if 'confidence' in all_predictions_df.columns:
            plt.figure(figsize=(10, 6))
            all_predictions_df['confidence'] = pd.to_numeric(all_predictions_df['confidence'], errors='coerce')
            all_predictions_df.dropna(subset=['confidence'], inplace=True)

            if not all_predictions_df.empty:
                confidence_bins = np.linspace(0, 1, 11)
                all_predictions_df['confidence_bin'] = pd.cut(all_predictions_df['confidence'],
                                                              bins=confidence_bins,
                                                              include_lowest=True,
                                                              right=True)

                conf_accuracy = all_predictions_df.groupby('confidence_bin')['is_correct'].mean()
                conf_counts = all_predictions_df.groupby('confidence_bin').size()

                ax = conf_accuracy.plot(kind='bar', color='coral')
                plt.title('Accuracy by Prediction Confidence Level')
                plt.xlabel('Confidence Range')
                plt.ylabel('Accuracy')
                plt.ylim(0, 1.05)
                ax.set_xticklabels([f"{inter.left:.1f}-{inter.right:.1f}]" for inter in conf_accuracy.index],
                                   rotation=45, ha="right")

                for i, acc in enumerate(conf_accuracy):
                    if pd.notna(acc):
                        plt.text(i, acc + 0.02, f'{acc:.2f}\n(n={conf_counts.iloc[i]})',
                                 ha='center', va='bottom')

                plt.tight_layout()
                plt.savefig('accuracy_by_confidence.png')
                plt.close()

        
        if 'initial_p2_revealed_count' in all_predictions_df.columns:
            revealed_order_pivot = all_predictions_df.groupby(['initial_p2_revealed_count', 'propagation_order'])['is_correct'].agg(['mean', 'count']).reset_index()

            if not revealed_order_pivot.empty:
                revealed_accuracy_pivot = revealed_order_pivot.pivot(
                    index='initial_p2_revealed_count',
                    columns='propagation_order',
                    values='mean'
                )

                plt.figure(figsize=(12, 8))
                sns.heatmap(revealed_accuracy_pivot, annot=True, cmap='YlGnBu', fmt='.3f', cbar_kws={'label': 'Accuracy'})
                plt.title('Prediction Accuracy by Initial Revealed Count and Propagation Order')
                plt.xlabel('Propagation Order')
                plt.ylabel('Initial P2 Pokémon Revealed Count')
                plt.tight_layout()
                plt.savefig('accuracy_by_revealed_and_order.png')
                plt.close()

                revealed_count_pivot = revealed_order_pivot.pivot(
                    index='initial_p2_revealed_count',
                    columns='propagation_order',
                    values='count'
                )

                plt.figure(figsize=(12, 8))
                sns.heatmap(revealed_count_pivot, annot=True, cmap='Greens', fmt='g', cbar_kws={'label': 'Number of Predictions'})
                plt.title('Number of Predictions by Initial Revealed Count and Propagation Order')
                plt.xlabel('Propagation Order')
                plt.ylabel('Initial P2 Pokémon Revealed Count')
                plt.tight_layout()
                plt.savefig('counts_by_revealed_and_order.png')
                plt.close()

    
    print("\n--- Evaluation Summary ---")
    print(f"Overall Accuracy: {results.get('overall', {}).get('accuracy', 0.0):.4f} " +
          f"(Correct: {results.get('overall', {}).get('correct',0)}, " +
          f"Total: {results.get('overall', {}).get('total',0)})")

    print("\nAccuracy by P2 Pokémon Slot Predicted:")
    for idx in range(1, 7):
        slot_key = f'pokemon_{idx}'
        slot_results = results.get(slot_key, {'accuracy': 0.0, 'correct': 0, 'total': 0})
        print(f"  P2 Slot {idx}: {slot_results['accuracy']:.4f} " +
              f"(Correct: {slot_results['correct']}, Total: {slot_results['total']})")

        if slot_results.get('by_propagation_order'):
            print(f"    By Propagation Order for Slot {idx}:")
            for order, order_data in sorted(slot_results['by_propagation_order'].items()):
                print(f"      Order {order}: {order_data['accuracy']:.4f} " +
                      f"(Correct: {order_data['correct']}, Total: {order_data['total']})")

    if not all_predictions_df.empty:
        print("\nAccuracy by Propagation Order (All Slots):")
        propagation_accuracy = all_predictions_df.groupby('propagation_order')['is_correct'].agg(['mean', 'count'])
        for order, data in propagation_accuracy.iterrows():
            print(f"  Order {order}: {data['mean']:.4f} (Total Predictions: {data['count']})")

def debug_prediction_process(models, df_full_dataset):
    """Run a detailed debug analysis of the prediction process for a single example."""
    print("\n===== DEBUGGING PREDICTION PROCESS WITH SEQUENCE DATA =====")

    
    sample_initial_state_row = None

    
    game_pokemon_counts = {}
    for game_id, game_df in df_full_dataset.groupby('game_id'):
        
        max_revealed = game_df['p2_number_of_pokemon_revealed'].max()
        game_pokemon_counts[game_id] = max_revealed

    
    good_games = [game_id for game_id, count in game_pokemon_counts.items() if count >= 4]

    if good_games:
        
        for revealed_count in range(1, 3):
            candidate_rows = df_full_dataset[
                (df_full_dataset['game_id'].isin(good_games)) &
                (df_full_dataset['p2_number_of_pokemon_revealed'] == revealed_count)
                ]

            if not candidate_rows.empty:
                sample_initial_state_row = candidate_rows.iloc[0]
                print(f"Selected a sample row (Game ID: {sample_initial_state_row['game_id']}, " +
                      f"Turn: {sample_initial_state_row['turn_id']}) " +
                      f"with {revealed_count} P2 Pokemon revealed")
                break

    if sample_initial_state_row is None:
        
        sample_initial_state_row = df_full_dataset.iloc[0]
        print(f"Using first row of dataset as sample " +
              f"(Game ID: {sample_initial_state_row['game_id']}, " +
              f"Turn: {sample_initial_state_row['turn_id']}). " +
              f"P2 revealed: {sample_initial_state_row['p2_number_of_pokemon_revealed']}")

    initial_state_dict = sample_initial_state_row.to_dict()

    
    print("\nSequence of P2 Pokemon reveals for this state:")
    initial_revealed_count = sample_initial_state_row['p2_number_of_pokemon_revealed']


    for i in range(1, initial_revealed_count + 1):
        pokemon_name = sample_initial_state_row.get(f'p2_pokemon{i}_name')
        print(f"  Already Revealed #{i}: {pokemon_name}")

    for i in range(1, 7 - initial_revealed_count):
        reveal_col = f'p2_next_reveal_{i}'
        if reveal_col in sample_initial_state_row and not pd.isna(sample_initial_state_row[reveal_col]):
            print(f"  Future Reveal #{i}: {sample_initial_state_row[reveal_col]}")

    
    model_start_idx = max(2, initial_revealed_count + 1)
    print(f"\nStarting propagation to predict from P2 Slot {model_start_idx} onwards.")

    prediction_chain, viz_data = propagate_predictions(initial_state_dict, models, model_start_idx)

    print("\n--- Detailed Prediction Chain with Sequence Evaluation ---")

    for step_idx, step_result in enumerate(prediction_chain):
        slot = step_result['pokemon_idx']
        order = step_result['propagation_order']

        if not step_result['predictions']:
            print(f"\nStep {step_idx+1} (Propagation Order: {order}): No predictions for P2 Slot {slot}")
            continue

        predicted_name = step_result['predictions'][0][0]
        confidence = step_result['predictions'][0][1]

        
        sequence_col = f'p2_next_reveal_{order + 1}'
        actual_name = sample_initial_state_row.get(sequence_col, "Unknown/Not in sequence")

        print(f"\nStep {step_idx+1} (Propagation Order: {order}): Predicting P2 Slot {slot}")
        print(f"  Model Predicted: {predicted_name} (Confidence: {confidence:.4f})")
        print(f"  Actual from Sequence Data: {actual_name}")
        print(f"  Correct: {predicted_name == actual_name if actual_name != 'Unknown/Not in sequence' else 'N/A'}")

    
    if viz_data['nodes']:
        output_viz_path = f"debug_prediction_game_{sample_initial_state_row['game_id']}_turn_{sample_initial_state_row['turn_id']}_with_sequence.png"
        create_prediction_visualization(viz_data, output_viz_path)

    print("\n===== DEBUG COMPLETE =====")
    return prediction_chain

def sample_data_stratified(df, sample_size=100, strat_col='p2_number_of_pokemon_revealed', random_state=42):
    """Sample data with stratification by number of revealed Pokemon."""
    if strat_col not in df.columns:
        print(f"Stratification column '{strat_col}' not found. Returning random sample or full df.")
        return df.sample(min(sample_size, len(df)), random_state=random_state) if sample_size else df

    if len(df) < sample_size or df[strat_col].nunique() == 1:
        return df.sample(min(sample_size, len(df)), random_state=random_state) if sample_size else df

    try:
        
        weights = df[strat_col].value_counts(normalize=True).sort_index().apply(lambda x: 1 / (x + 0.1))

        sampled_df = df.groupby(strat_col, group_keys=False).apply(
            lambda x: x.sample(
                n=max(1, min(len(x), int(np.ceil(sample_size * (len(x)/len(df)) * weights.get(x.name, 1))))),
                random_state=random_state, replace=False)
        ).reset_index(drop=True)

        
        if len(sampled_df) < sample_size * 0.8 and len(df) > sample_size:
            additional_samples_needed = sample_size - len(sampled_df)
            if additional_samples_needed > 0:
                remaining_df = df.drop(sampled_df.index, errors='ignore')
                if len(remaining_df) >= additional_samples_needed:
                    sampled_df = pd.concat([sampled_df, remaining_df.sample(additional_samples_needed, random_state=random_state)])
                else:
                    sampled_df = pd.concat([sampled_df, remaining_df])

        if len(sampled_df) > sample_size * 1.2:
            sampled_df = sampled_df.sample(sample_size, random_state=random_state)

    except Exception as e:
        print(f"Stratified sampling failed: {e}. Returning random sample.")
        sampled_df = df.sample(min(sample_size, len(df)), random_state=random_state)

    print(f"Sampled data shape: {sampled_df.shape}")
    print(f"Distribution of '{strat_col}' in sampled data:\n{sampled_df[strat_col].value_counts(normalize=True).sort_index()}")
    return sampled_df

def create_sample_visualizations(df, models, num_samples_per_category=1):
    """Create prediction visualizations for different starting positions."""
    if 'game_id' not in df.columns or 'turn_id' not in df.columns:
        print("Cannot create sample visualizations: 'game_id' or 'turn_id' missing from DataFrame.")
        return

    
    for revealed_count in range(0, 6):
        
        example_rows = df[
            (df['p2_number_of_pokemon_revealed'] == revealed_count) &
            (df['p2_next_reveal_1'].notna())  
            ]

        if example_rows.empty:
            print(f"\nNo samples found with {revealed_count} P2 Pokémon initially revealed and future reveal data.")
            continue

        
        for i, (_, example_row_series) in enumerate(example_rows.sample(
                min(num_samples_per_category, len(example_rows))).iterrows()):
            if i >= num_samples_per_category:
                break

            example_row_dict = example_row_series.to_dict()
            game_id = example_row_series.get('game_id', 'UnknownGame')
            turn_id = example_row_series.get('turn_id', 'UnknownTurn')

            first_slot_to_predict = revealed_count + 1
            model_start_idx = max(2, first_slot_to_predict)

            print(f"\nCreating prediction visualization for Game {game_id}, Turn {turn_id} " +
                  f"(Initial P2 Revealed: {revealed_count}, Predicting from Slot {model_start_idx})")

            
            print("Future Pokemon reveals in this game state:")
            for seq_idx in range(1, 7 - revealed_count):
                seq_col = f'p2_next_reveal_{seq_idx}'
                if seq_col in example_row_series and not pd.isna(example_row_series[seq_col]):
                    print(f"  Next reveal #{seq_idx}: {example_row_series[seq_col]}")

            if model_start_idx > 6:
                print("  All 6 Pokémon already revealed or accounted for. Skipping visualization.")
                continue

            try:
                _, viz_data = propagate_predictions(example_row_dict, models, model_start_idx)
                if viz_data['nodes']:
                    viz_filename = f"prediction_viz_game_{game_id}_turn_{turn_id}_start_slot{model_start_idx}_with_sequence.png"
                    create_prediction_visualization(viz_data, viz_filename)
                else:
                    print("  No visualization data generated by propagate_predictions.")
            except Exception as e:
                print(f"  Failed to create visualization: {e}")

def main():
    """Main function to run the Pokemon prediction evaluation."""
    print("Loading data...")
    df = pd.read_csv(FILE_PATH)

    if 'game_id' not in df.columns:
        print(f"CRITICAL ERROR: 'game_id' column not found in {FILE_PATH}.")
        print("This script requires 'game_id' to correctly evaluate propagated predictions.")
        return

    
    sequence_columns = [col for col in df.columns if 'next_reveal' in col]
    if not sequence_columns:
        print(f"CRITICAL ERROR: No 'p2_next_reveal_N' columns found in {FILE_PATH}.")
        print("Please run the create_pokemon_sequence_csv.py script first to generate the sequence data.")
        return

    print(f"Found sequence columns: {', '.join(sequence_columns)}")

    
    if SAMPLE_SIZE is not None and SAMPLE_SIZE > 0 and SAMPLE_SIZE < len(df):
        print(f"\nSampling data to {SAMPLE_SIZE} rows...")
        df = sample_data_stratified(df, SAMPLE_SIZE, strat_col='p2_number_of_pokemon_revealed')
    else:
        print("\nUsing full dataset or SAMPLE_SIZE is not applicable.")

    print("\nLoading models...")
    try:
        models = load_latest_models()
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    
    if DEBUG_MODE:
        try:
            debug_prediction_process(models, df)
        except Exception as e:
            print(f"Error during debug_prediction_process: {e}")

    
    print("\nCreating sample visualizations...")
    try:
        create_sample_visualizations(df, models, num_samples_per_category=1)
    except Exception as e:
        print(f"Error during create_sample_visualizations: {e}")

    
    print("\nEvaluating models using the sequence-based evaluation approach...")
    try:
        evaluation_results, all_detailed_preds_log = evaluate_predictions_with_sequences(df, models)
        all_predictions_df = pd.DataFrame(all_detailed_preds_log)
    except Exception as e:
        print(f"Error during model evaluation: {e}")
        return

    
    if not all_predictions_df.empty:
        print("\nVisualizing results...")
        try:
            visualize_results(evaluation_results, all_predictions_df)
        except Exception as e:
            print(f"Error during results visualization: {e}")
    else:
        print("\nNo detailed predictions were logged; skipping result visualization.")

    print("\nEvaluation complete. Check the output files for visualizations and logs.")

if __name__ == "__main__":
    main()