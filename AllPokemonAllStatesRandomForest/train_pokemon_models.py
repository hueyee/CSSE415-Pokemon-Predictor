import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time
import datetime
import os
import warnings
warnings.filterwarnings('ignore')

FILE_PATH = "../Parquets/all_pokemon_moves.csv"
MODELS_DIR = "../Models"
TARGET_PREFIX = "p2_pokemon"
TARGET_SUFFIX = "_name"
USE_RATING_FEATURES = True
USE_CURRENT_POKEMON = True
USE_PREVIOUS_POKEMON = True
USE_POKEMON_COUNT = True
USE_MOVES = True
USE_TURN_INFO = True
N_ESTIMATORS = 1000
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
TEST_SIZE = 0.2
SHOW_FEATURE_IMPORTANCE = True
TOP_N_FEATURES = 20

def load_data(file_path):
    print(f"Loading data from {file_path}...")
    return pd.read_csv(file_path)

def preprocess_data(df, pokemon_idx):
    print(f"Preprocessing data for Pokemon {pokemon_idx}...")
    processed_df = df.copy()

    target_column = f"next_pokemon"

    features = []
    if USE_RATING_FEATURES:
        features.extend(['p1_rating', 'p2_rating'])

    if USE_CURRENT_POKEMON:
        features.extend(['p1_current_pokemon', 'p2_current_pokemon'])

    if USE_PREVIOUS_POKEMON:
        features.extend([
            'p1_pokemon1_name', 'p1_pokemon2_name', 'p1_pokemon3_name', 'p1_pokemon4_name', 'p1_pokemon5_name',
            'p1_pokemon6_name',
        ])

        for i in range(1, pokemon_idx):
            features.append(f'p2_pokemon{i}_name')

    if USE_POKEMON_COUNT:
        features.extend(['p1_number_of_pokemon_revealed', 'p2_number_of_pokemon_revealed'])

    if USE_MOVES:
        for i in range(1, 7):
            for j in range(1, 5):
                features.append(f'p1_pokemon{i}_move{j}')

        for i in range(1, pokemon_idx):
            for j in range(1, 5):
                features.append(f'p2_pokemon{i}_move{j}')

    if USE_TURN_INFO:
        features.extend(['turn_id'])

    mask = processed_df['p2_number_of_pokemon_revealed'] >= pokemon_idx - 1
    processed_df = processed_df[mask]

    processed_df = processed_df.dropna(subset=[target_column])

    categorical_features = []
    numerical_features = []

    for feature in features:
        if feature in processed_df.columns:
            if processed_df[feature].dtype == 'object':
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

    return X, y, categorical_features, numerical_features, processed_df

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

def train_model(X, y, categorical_features, numerical_features, pokemon_idx):
    print(f"Training model for Pokemon {pokemon_idx}...")
    print("Splitting data into train and test sets...")

    class_counts = y.value_counts()
    rare_classes = class_counts[class_counts < 2].index
    if len(rare_classes) > 0:
        print(f"Removing {len(rare_classes)} rare Pokemon classes with only 1 occurrence")
        keep_mask = ~y.isin(rare_classes)
        X = X[keep_mask]
        y = y[keep_mask]
        print(f"Data shape after removing rare classes: {X.shape}")

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
    X_train_num = X_train[numerical_features].values if numerical_features else np.zeros((X_train.shape[0], 0))
    X_test_num = X_test[numerical_features].values if numerical_features else np.zeros((X_test.shape[0], 0))

    print("Combining features...")
    X_train_combined = np.hstack([X_train_cat, X_train_num])
    X_test_combined = np.hstack([X_test_cat, X_test_num])

    feature_names = encoder.get_feature_names() + numerical_features

    print(f"Creating and training random forest with {N_ESTIMATORS} trees...")
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
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_pred,
        'X_train_combined': X_train_combined,
        'X_test_combined': X_test_combined
    }

    return model_info

def evaluate_model(model_info, pokemon_idx):
    print(f"\nModel Evaluation for Pokemon {pokemon_idx}:")
    model = model_info['model']
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

    all_classes = sorted(list(set(y_test.unique()).union(set(y_pred))))
    print(f"\nConfusion Matrix (all {len(all_classes)} classes):")
    cm = confusion_matrix(
        y_test, y_pred,
        labels=all_classes
    )
    cm_df = pd.DataFrame(cm, index=all_classes, columns=all_classes)

    try:
        plt.figure(figsize=(2560/300, 1440/300), dpi=300)
        sns.heatmap(cm_df, annot=False, fmt='d', cmap='RdBu_r', xticklabels=False, yticklabels=False)
        plt.title(f'Confusion Matrix - Pokemon {pokemon_idx}')
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_pokemon_{pokemon_idx}.png', dpi=300)
        plt.close()
        print(f"\nConfusion matrix plot saved as 'confusion_matrix_pokemon_{pokemon_idx}.png'")
    except Exception as e:
        print(f"Could not create confusion matrix plot: {e}")

    if SHOW_FEATURE_IMPORTANCE:
        show_feature_importance(model_info, pokemon_idx)

def show_feature_importance(model_info, pokemon_idx):
    print(f"\nFeature Importance Analysis for Pokemon {pokemon_idx}:")
    model = model_info['model']
    feature_names = model_info['feature_names']

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
    for i, row in base_importance.head(TOP_N_FEATURES).iterrows():
        print(f"{i+1}. {row['Base_Feature']}: {row['Importance']:.4f}")

    print(f"\nTop {TOP_N_FEATURES} Individual Feature Values:")
    for i, row in importance_df.head(TOP_N_FEATURES).iterrows():
        print(f"{i+1}. {row['Feature']}: {row['Importance']:.4f}")

    try:
        plt.figure(figsize=(12, 10))
        plt.title(f'Aggregate Feature Importance - Pokemon {pokemon_idx}')
        top_n = min(15, len(base_importance))
        sns.barplot(x='Importance', y='Base_Feature', data=base_importance.head(top_n))
        plt.tight_layout()
        plt.savefig(f'aggregate_feature_importance_pokemon_{pokemon_idx}.png')
        plt.close()
        print(f"\nAggregate feature importance plot saved as 'aggregate_feature_importance_pokemon_{pokemon_idx}.png'")
    except Exception as e:
        print(f"Could not create aggregate feature importance plot: {e}")

    try:
        plt.figure(figsize=(14, 20))
        plt.title(f'Top {TOP_N_FEATURES} Individual Feature Importances - Pokemon {pokemon_idx}')
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(TOP_N_FEATURES))
        plt.tight_layout()
        plt.savefig(f'individual_feature_importance_pokemon_{pokemon_idx}.png')
        plt.close()
        print(f"Top {TOP_N_FEATURES} individual feature importance plot saved as 'individual_feature_importance_pokemon_{pokemon_idx}.png'")
    except Exception as e:
        print(f"Could not create individual feature importance plot: {e}")

def save_model_package(model_info, output_dir, pokemon_idx):
    model_filename = os.path.join(output_dir, f"pokemon_prediction_model_{pokemon_idx}.joblib")
    print(f"\nSaving model package to '{model_filename}'...")

    model_package = {
        'model': model_info['model'],
        'encoder': model_info['encoder'],
        'feature_names': model_info['feature_names'],
        'categorical_features': model_info['categorical_features'],
        'numerical_features': model_info['numerical_features'],
        'pokemon_idx': pokemon_idx
    }

    joblib.dump(model_package, model_filename)
    print(f"Model package saved to '{model_filename}'")

def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(MODELS_DIR, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Models will be saved to: {output_dir}")

    df = load_data(FILE_PATH)

    for pokemon_idx in range(2, 7):
        print(f"\n{'='*80}")
        print(f"Processing Pokemon {pokemon_idx}")
        print(f"{'='*80}\n")

        X, y, categorical_features, numerical_features, processed_df = preprocess_data(df, pokemon_idx)
        model_info = train_model(X, y, categorical_features, numerical_features, pokemon_idx)
        evaluate_model(model_info, pokemon_idx)
        save_model_package(model_info, output_dir, pokemon_idx)

if __name__ == "__main__":
    main()