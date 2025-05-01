import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

FILE_PATH = "parsed_showdown_replays.parquet"
TARGET_COLUMN = 'p2_revealed_pokemon'
USE_RATING_FEATURES = True
USE_CURRENT_POKEMON = True
USE_MOVES = True
USE_DAMAGE_INFO = True
USE_STATUS_INFO = True
USE_OPPONENT_REVEALED = True
USE_TURN_INFO = True
USE_INTERACTION_FEATURES = True
USE_DERIVED_FEATURES = True
N_ESTIMATORS = 2000
MAX_DEPTH = 50
MIN_SAMPLES_SPLIT = 2
MIN_SAMPLES_LEAF = 1
MAX_FEATURES = 1/3
BOOTSTRAP = True
CLASS_WEIGHT = 'balanced_subsample'
RANDOM_STATE = 42
CRITERION = 'entropy'
N_JOBS = -1
WARM_START = True
OOB_SCORE = True
CV_FOLDS = 5
SHOW_FEATURE_IMPORTANCE = True
TOP_N_FEATURES = 50
TEST_SIZE = 0.2

def load_data(file_path):
    print(f"Loading data from {file_path}...")
    return pd.read_parquet(file_path)

def create_derived_features(df):
    print("Creating derived features...")
    df_with_features = df.copy()
    df_with_features['matchup'] = df_with_features['p1_pokemon'] + '_vs_' + df_with_features['p2_pokemon']
    df_with_features['rating_diff'] = df_with_features['p1_rating'] - df_with_features['p2_rating']
    df_with_features['rating_advantage'] = (df_with_features['rating_diff'] > 0).astype(int)

    def get_game_stage(turn_id):
        if turn_id <= 2:
            return 'early'
        elif turn_id <= 10:
            return 'mid'
        else:
            return 'late'
    df_with_features['game_stage'] = df_with_features['turn_id'].apply(get_game_stage)

    top_pokemon = ['Skarmory', 'Blissey', 'Tyranitar', 'Swampert', 'Gengar',
                   'Metagross', 'Salamence', 'Celebi', 'Zapdos', 'Magneton']
    for pokemon in top_pokemon:
        df_with_features[f'p1_revealed_{pokemon}'] = (df_with_features['p1_revealed_pokemon'] == pokemon).astype(int)
        df_with_features[f'p2_has_{pokemon}'] = (df_with_features['p2_pokemon'] == pokemon).astype(int)

    status_types = ['par', 'tox', 'slp', 'brn', 'frz', 'psn']
    for status in status_types:
        df_with_features[f'p1_status_{status}'] = (df_with_features['p1_status'] == status).astype(int)
        df_with_features[f'p2_status_{status}'] = (df_with_features['p2_status'] == status).astype(int)

    df_with_features['p1_has_status_effect'] = (~df_with_features['p1_status'].isna()).astype(int)
    df_with_features['p2_has_status_effect'] = (~df_with_features['p2_status'].isna()).astype(int)

    p1_damage = df_with_features['p1_damage_taken'].fillna(0)
    p2_damage = df_with_features['p2_damage_taken'].fillna(0)
    df_with_features['damage_diff'] = p2_damage - p1_damage
    df_with_features['p1_advantage'] = (df_with_features['damage_diff'] > 0).astype(int)
    df_with_features['equal_damage'] = (df_with_features['damage_diff'] == 0).astype(int)

    setup_moves = ['Swords Dance', 'Dragon Dance', 'Calm Mind', 'Agility', 'Bulk Up', 'Curse']
    status_moves = ['Thunder Wave', 'Toxic', 'Will-O-Wisp', 'Hypnosis', 'Spore', 'Stun Spore']
    defensive_moves = ['Protect', 'Substitute', 'Recover', 'Rest', 'Wish', 'Reflect', 'Light Screen']
    df_with_features['p1_used_setup'] = df_with_features['p1_move'].isin(setup_moves).astype(int)
    df_with_features['p1_used_status'] = df_with_features['p1_move'].isin(status_moves).astype(int)
    df_with_features['p1_used_defensive'] = df_with_features['p1_move'].isin(defensive_moves).astype(int)
    df_with_features['p2_used_setup'] = df_with_features['p2_move'].isin(setup_moves).astype(int)
    df_with_features['p2_used_status'] = df_with_features['p2_move'].isin(status_moves).astype(int)
    df_with_features['p2_used_defensive'] = df_with_features['p2_move'].isin(defensive_moves).astype(int)

    df_with_features['battle_state'] = df_with_features['game_stage'] + '_' + \
                                       df_with_features['p1_advantage'].astype(str) + '_' + \
                                       df_with_features['p1_has_status_effect'].astype(str)

    df_with_features['has_revealed_defensive_pokemon'] = df_with_features['p2_revealed_pokemon'].isin(
        ['Skarmory', 'Blissey', 'Snorlax', 'Umbreon', 'Suicune']).astype(int)
    df_with_features['has_revealed_offensive_pokemon'] = df_with_features['p2_revealed_pokemon'].isin(
        ['Tyranitar', 'Salamence', 'Metagross', 'Gengar', 'Zapdos', 'Swampert']).astype(int)

    df_with_features['turn_squared'] = df_with_features['turn_id'] ** 2
    df_with_features['turn_log'] = np.log1p(df_with_features['turn_id'])

    df_with_features['rating_sum'] = df_with_features['p1_rating'] + df_with_features['p2_rating']
    df_with_features['rating_product'] = df_with_features['p1_rating'] * df_with_features['p2_rating']
    return df_with_features

def preprocess_data(df, target_column):
    print("Preprocessing data...")
    processed_df = df.copy()

    if USE_DERIVED_FEATURES:
        processed_df = create_derived_features(processed_df)

    features = []

    if USE_RATING_FEATURES:
        features.extend(['p1_rating', 'p2_rating'])
        if USE_DERIVED_FEATURES:
            features.extend(['rating_diff', 'rating_advantage', 'rating_sum', 'rating_product'])

    if USE_CURRENT_POKEMON:
        features.extend(['p1_pokemon', 'p2_pokemon'])
        if USE_DERIVED_FEATURES:
            features.extend(['matchup'])
            for pokemon in ['Skarmory', 'Blissey', 'Tyranitar', 'Swampert', 'Gengar',
                            'Metagross', 'Salamence', 'Celebi', 'Zapdos', 'Magneton']:
                features.extend([f'p1_revealed_{pokemon}', f'p2_has_{pokemon}'])

    if USE_MOVES:
        features.extend(['p1_move', 'p2_move'])
        if USE_DERIVED_FEATURES:
            features.extend([
                'p1_used_setup', 'p1_used_status', 'p1_used_defensive',
                'p2_used_setup', 'p2_used_status', 'p2_used_defensive'
            ])

    if USE_DAMAGE_INFO:
        features.extend(['p1_damage_taken', 'p2_damage_taken'])
        if USE_DERIVED_FEATURES:
            features.extend(['damage_diff', 'p1_advantage', 'equal_damage'])

    if USE_STATUS_INFO:
        features.extend(['p1_status', 'p2_status'])
        if USE_DERIVED_FEATURES:
            status_types = ['par', 'tox', 'slp', 'brn', 'frz', 'psn']
            for status in status_types:
                features.extend([f'p1_status_{status}', f'p2_status_{status}'])
            features.extend(['p1_has_status_effect', 'p2_has_status_effect'])

    if USE_OPPONENT_REVEALED:
        features.extend(['p1_revealed_pokemon'])
        if USE_DERIVED_FEATURES:
            features.extend(['has_revealed_defensive_pokemon', 'has_revealed_offensive_pokemon'])

    if USE_TURN_INFO:
        features.extend(['turn_id'])
        if USE_DERIVED_FEATURES:
            features.extend(['game_stage', 'turn_squared', 'turn_log'])

    if USE_DERIVED_FEATURES and USE_DAMAGE_INFO and USE_STATUS_INFO and USE_TURN_INFO:
        features.extend(['battle_state'])

    processed_df = processed_df.dropna(subset=[target_column])

    categorical_features = []
    numerical_features = []
    for feature in features:
        if feature in processed_df.columns:
            if (processed_df[feature].dtype == 'object' or
                    feature in [
                        'p1_pokemon', 'p2_pokemon', 'p1_move', 'p2_move',
                        'p1_status', 'p2_status', 'p1_revealed_pokemon', 'matchup',
                        'game_stage', 'battle_state'
                    ]):
                categorical_features.append(feature)
            else:
                numerical_features.append(feature)

    for col in categorical_features:
        if col in processed_df.columns:
            processed_df[col] = processed_df[col].fillna('Unknown')

    for col in numerical_features:
        if col in processed_df.columns:
            processed_df[col] = processed_df[col].fillna(processed_df[col].median())

    X = processed_df[features]
    y = processed_df[target_column]

    print(f"Final preprocessed data shape: {X.shape}")
    print(f"Number of categorical features: {len(categorical_features)}")
    print(f"Number of numerical features: {len(numerical_features)}")
    print(f"Target distribution sample:")
    print(y.value_counts().head(10))

    feature_mapping = {}
    cat_start_idx = 0
    for i, cat_feature in enumerate(categorical_features):
        unique_values = X[cat_feature].unique()
        feature_mapping[cat_feature] = {
            'type': 'categorical',
            'values': {value: cat_start_idx + j for j, value in enumerate(unique_values)},
            'start_idx': cat_start_idx,
            'end_idx': cat_start_idx + len(unique_values) - 1
        }
        cat_start_idx += len(unique_values)

    for i, num_feature in enumerate(numerical_features):
        feature_mapping[num_feature] = {
            'type': 'numerical',
            'index': cat_start_idx + i
        }
    return X, y, categorical_features, numerical_features, processed_df, feature_mapping

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

def train_model_with_tracking(X, y, categorical_features, numerical_features, feature_mapping):
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    print("Creating and fitting custom one-hot encoder...")
    encoder = CustomOneHotEncoder().fit(X_train, categorical_features)

    print("Transforming categorical features...")
    X_train_cat = encoder.transform(X_train, categorical_features)
    X_test_cat = encoder.transform(X_test, categorical_features)

    print("Extracting numerical features...")
    X_train_num = X_train[numerical_features].values
    X_test_num = X_test[numerical_features].values

    print("Combining features...")
    X_train_combined = np.hstack([X_train_cat, X_train_num])
    X_test_combined = np.hstack([X_test_cat, X_test_num])

    feature_names = encoder.get_feature_names() + numerical_features

    print(f"Creating and training random forest with {N_ESTIMATORS} trees...")
    print("This will take a while...")

    start_time = time.time()

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features=MAX_FEATURES,
        bootstrap=BOOTSTRAP,
        class_weight=CLASS_WEIGHT,
        criterion=CRITERION,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        warm_start=WARM_START,
        oob_score=OOB_SCORE,
        verbose=1
    )

    trees_per_batch = 200
    for i in range(0, N_ESTIMATORS, trees_per_batch):
        batch_end = min(i + trees_per_batch, N_ESTIMATORS)
        print(f"Training trees {i+1} to {batch_end}...")
        model.n_estimators = batch_end
        model.fit(X_train_combined, y_train)

        elapsed_time = time.time() - start_time
        print(f"  Trained {batch_end} trees in {elapsed_time:.2f} seconds.")
        if OOB_SCORE:
            print(f"  Current out-of-bag score: {model.oob_score_:.4f}")

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds.")

    print("Making predictions on test set...")
    y_pred = model.predict(X_test_combined)

    model_info = {
        'model': model,
        'encoder': encoder,
        'feature_names': feature_names,
        'categorical_features': categorical_features,
        'numerical_features': numerical_features,
        'feature_mapping': feature_mapping,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_pred,
        'X_train_combined': X_train_combined,
        'X_test_combined': X_test_combined
    }
    return model_info

def show_interpretable_feature_importance(model_info):
    print("\nFeature Importance Analysis:")

    model = model_info['model']
    feature_names = model_info['feature_names']
    feature_mapping = model_info['feature_mapping']

    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })

    importance_df = importance_df.sort_values('Importance', ascending=False).reset_index(drop=True)

    importance_df['Base_Feature'] = importance_df['Feature'].apply(
        lambda x: x.split('_')[0] if '_' in x else x
    )

    base_importance = importance_df.groupby('Base_Feature')['Importance'].sum().reset_index()
    base_importance = base_importance.sort_values('Importance', ascending=False).reset_index(drop=True)

    print("\nAggregate Feature Importance (by original column):")
    for i, row in base_importance.iterrows():
        print(f"{i+1}. {row['Base_Feature']}: {row['Importance']:.4f}")

    print(f"\nTop {TOP_N_FEATURES} Individual Feature Values:")
    for i, row in importance_df.head(TOP_N_FEATURES).iterrows():
        print(f"{i+1}. {row['Feature']}: {row['Importance']:.4f}")

    try:
        plt.figure(figsize=(12, 10))
        plt.title('Aggregate Feature Importance')
        top_n = min(15, len(base_importance))
        sns.barplot(x='Importance', y='Base_Feature', data=base_importance.head(top_n))
        plt.tight_layout()
        plt.savefig('aggregate_feature_importance.png')
        plt.close()
        print("\nAggregate feature importance plot saved as 'aggregate_feature_importance.png'")
    except Exception as e:
        print(f"Could not create aggregate feature importance plot: {e}")

    try:
        plt.figure(figsize=(14, 20))
        plt.title(f'Top {TOP_N_FEATURES} Individual Feature Importances')
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(TOP_N_FEATURES))
        plt.tight_layout()
        plt.savefig('individual_feature_importance.png')
        plt.close()
        print(f"Top {TOP_N_FEATURES} individual feature importance plot saved as 'individual_feature_importance.png'")
    except Exception as e:
        print(f"Could not create individual feature importance plot: {e}")

    print("\nPokemon-Specific Feature Importance:")

    top_pokemon = [
        'Skarmory', 'Blissey', 'Tyranitar', 'Swampert', 'Gengar',
        'Metagross', 'Salamence', 'Celebi', 'Zapdos', 'Magneton'
    ]
    for pokemon in top_pokemon:
        pokemon_features = importance_df[importance_df['Feature'].str.contains(pokemon)]
        if not pokemon_features.empty:
            total_importance = pokemon_features['Importance'].sum()
            print(f"\n{pokemon} total importance: {total_importance:.4f}")
            for i, row in pokemon_features.iterrows():
                print(f"  {row['Feature']}: {row['Importance']:.4f}")

def evaluate_model(model_info):
    print("\nModel Evaluation:")

    model = model_info['model']
    X_test_combined = model_info['X_test_combined']
    y_test = model_info['y_test']
    y_pred = model_info['y_pred']

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    if hasattr(model, 'oob_score_'):
        print(f"Out-of-bag score: {model.oob_score_:.4f}")

    print("\nTop 10 class prediction counts:")
    print(pd.Series(y_pred).value_counts().head(10))

    top_classes = y_test.value_counts().head(15).index.tolist()
    print("\nClassification Report (top 15 classes):")
    print(classification_report(y_test, y_pred, labels=top_classes, zero_division=0))

    top5_classes = y_test.value_counts().head(5).index.tolist()
    print("\nConfusion Matrix (top 5 classes):")
    cm = confusion_matrix(
        y_test, y_pred,
        labels=top5_classes
    )

    cm_df = pd.DataFrame(cm, index=top5_classes, columns=top5_classes)
    print(cm_df)

    try:
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()
        print("\nConfusion matrix plot saved as 'confusion_matrix.png'")
    except Exception as e:
        print(f"Could not create confusion matrix plot: {e}")

    if SHOW_FEATURE_IMPORTANCE:
        show_interpretable_feature_importance(model_info)

    print("\nError Analysis:")

    X_test_reset = model_info['X_test'].reset_index(drop=True)
    error_df = pd.DataFrame({
        'Actual': y_test.reset_index(drop=True),
        'Predicted': y_pred
    })

    for col in ['p1_pokemon', 'p2_pokemon', 'p1_revealed_pokemon', 'turn_id']:
        if col in X_test_reset.columns:
            error_df[col] = X_test_reset[col]

    error_df['Correct'] = (error_df['Actual'] == error_df['Predicted']).astype(int)

    by_pokemon = error_df.groupby('p2_pokemon').agg({
        'Correct': 'mean',
        'Actual': 'count'
    }).rename(columns={'Correct': 'Accuracy', 'Actual': 'Count'})
    by_pokemon = by_pokemon.sort_values('Count', ascending=False).head(10)
    print("\nAccuracy by Current Opponent Pokemon (top 10 by frequency):")
    print(by_pokemon)

    if 'turn_id' in error_df.columns:
        turn_mapping = {
            0: 'Turn 0',
            1: 'Turn 1',
            2: 'Turn 2',
            3: 'Turn 3-5',
            4: 'Turn 6-10',
            5: 'Turn 11+'
        }
        def map_turn(turn):
            if turn <= 2:
                return turn
            elif turn <= 5:
                return 3
            elif turn <= 10:
                return 4
            else:
                return 5
        error_df['turn_group'] = error_df['turn_id'].apply(map_turn)
        error_df['turn_label'] = error_df['turn_group'].map(turn_mapping)
        by_turn = error_df.groupby('turn_label').agg({
            'Correct': 'mean',
            'Actual': 'count'
        }).rename(columns={'Correct': 'Accuracy', 'Actual': 'Count'})
        print("\nAccuracy by Game Stage:")
        print(by_turn)

def get_prediction_probabilities(model_info, sample_data, top_n=5):
    model = model_info['model']
    encoder = model_info['encoder']
    categorical_features = model_info['categorical_features']
    numerical_features = model_info['numerical_features']

    sample_cat = encoder.transform(sample_data, categorical_features)
    sample_num = sample_data[numerical_features].values
    sample_combined = np.hstack([sample_cat, sample_num])

    proba = model.predict_proba(sample_combined)[0]
    classes = model.classes_

    sorted_indices = np.argsort(proba)[::-1][:top_n]
    top_classes = classes[sorted_indices]
    top_probas = proba[sorted_indices]
    return top_classes, top_probas

def save_model_package(model_info, filename="pokemon_prediction_model_package.joblib"):
    print(f"\nSaving model package to '{filename}'...")

    model_package = {
        'model': model_info['model'],
        'encoder': model_info['encoder'],
        'feature_names': model_info['feature_names'],
        'categorical_features': model_info['categorical_features'],
        'numerical_features': model_info['numerical_features'],
        'feature_mapping': model_info['feature_mapping']
    }
    joblib.dump(model_package, filename)
    print(f"Model package saved to '{filename}'")

def main():
    df = load_data(FILE_PATH)

    X, y, categorical_features, numerical_features, processed_df, feature_mapping = preprocess_data(df, TARGET_COLUMN)

    model_info = train_model_with_tracking(X, y, categorical_features, numerical_features, feature_mapping)

    evaluate_model(model_info)

    print("\nSample Prediction with Probabilities:")

    sample_idx = np.random.randint(0, len(model_info['X_test']))
    sample_data = model_info['X_test'].iloc[sample_idx:sample_idx+1]

    actual_pokemon = model_info['y_test'].iloc[sample_idx]

    top_predictions, top_probabilities = get_prediction_probabilities(model_info, sample_data, top_n=5)

    print(f"Sample input features:")
    for col in sample_data.columns:
        print(f"  {col}: {sample_data[col].values[0]}")
    print(f"Actual next Pokémon: {actual_pokemon}")
    print(f"Top 5 predictions:")
    for i, (pokemon, prob) in enumerate(zip(top_predictions, top_probabilities)):
        print(f"  {i+1}. {pokemon}: {prob:.4f} probability")

    save_model_package(model_info)

if __name__ == "__main__":
    main()