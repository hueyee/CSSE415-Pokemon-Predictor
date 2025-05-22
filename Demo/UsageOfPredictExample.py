import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
        return self.feature_names

from Predict import predict_team

print("=== Pokemon Team Prediction Examples ===\n")

print("Example 1: Just the first Pokemon (predict slots 2-6)")
result1 = predict_team(
    p1_rating=1500,
    p2_rating=1600,
    p2_pokemon1_name="Pikachu"
)
print("Predictions made - check prediction_results.png and detailed_predictions.png\n")

print("Example 2: First three Pokemon known (predict slots 4-6)")
result2 = predict_team(
    p1_rating=1800,
    p2_rating=1700,
    p2_pokemon1_name="Garchomp",
    p2_pokemon2_name="Rotom-Wash",
    p2_pokemon3_name="Clefable"
)
print("Predictions made - check prediction_results.png and detailed_predictions.png\n")

print("Example 3: High-level players with 4 Pokemon known (predict slots 5-6)")
result3 = predict_team(
    p1_rating=2000,
    p2_rating=1950,
    p2_pokemon1_name="Landorus-Therian",
    p2_pokemon2_name="Tapu Koko",
    p2_pokemon3_name="Ferrothorn",
    p2_pokemon4_name="Toxapex"
)
print("Predictions made - check prediction_results.png and detailed_predictions.png\n")

print("Example 4: First two Pokemon (predict slots 3-6)")
result4 = predict_team(
    p1_rating=1650,
    p2_rating=1700,
    p2_pokemon1_name="Dragonite",
    p2_pokemon2_name="Magnezone"
)
print("Predictions made - check prediction_results.png and detailed_predictions.png\n")

print("All examples complete! Check the generated PNG files for visual results.")
print("\nFunction signature:")
print("predict_team(p1_rating, p2_rating, p2_pokemon1_name,")
print("            p2_pokemon2_name=None, p2_pokemon3_name=None, p2_pokemon4_name=None,")
print("            p2_pokemon5_name=None, turn_id=1)")
