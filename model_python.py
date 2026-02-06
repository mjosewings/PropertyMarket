"""
Model Development and Evaluation Script (Python)
This script develops a logistic regression model to classify zip codes
based on house features. Visualizations are handled by R.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings

warnings.filterwarnings('ignore')


class HouseClassifier:
    def __init__(self, data_path):
        """Initialize the classifier with data"""
        self.df = pd.read_csv(data_path)
        self.model = None
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def prepare_data(self, test_size=0.25, random_state=42):
        """Prepare data for modeling"""
        print("\n" + "=" * 60)
        print("DATA PREPARATION")
        print("=" * 60)

        # Features and target
        X = self.df[['beds', 'baths', 'sqft', 'price']]
        y = self.df['zip_code']

        print(f"\nFeatures shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"Target classes: {y.unique()}")

        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print(f"\nTrain set size: {len(self.X_train)}")
        print(f"Test set size: {len(self.X_test)}")

        # Feature scaling
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        print("\nData preparation complete")
        print(f"Features scaled using StandardScaler")
        print(f"Train/Test split: {int((1 - test_size) * 100)}/{int(test_size * 100)}")

    def train_model(self):
        """Train logistic regression model with hyperparameter tuning"""
        print("\n" + "=" * 60)
        print("MODEL TRAINING")
        print("=" * 60)

        # Define parameter grid for GridSearch
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'solver': ['lbfgs', 'liblinear', 'saga'],
            'max_iter': [200, 500, 1000]
        }

        print("\nPerforming Grid Search for hyperparameter tuning...")
        print(f"Parameter grid: {param_grid}")

        # Create base model
        base_model = LogisticRegression(random_state=42, multi_class='multinomial')

        # Grid search with cross-validation
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(self.X_train_scaled, self.y_train)

        # Best model
        self.model = grid_search.best_estimator_

        print(f"\nBest parameters found: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

        # Cross-validation scores
        cv_scores = cross_val_score(self.model, self.X_train_scaled, self.y_train, cv=5)
        print(f"\nCross-Validation Scores: {cv_scores}")
        print(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")


# Create directories if they don't exist
import os
os.makedirs('data', exist_ok=True)

# Initialize classifier
classifier = HouseClassifier('data/house_data.csv')

# Prepare data
classifier.prepare_data(test_size=0.25, random_state=42)

# Train model
classifier.train_model()

# Evaluate model (without creating plots - those will be done in R)
print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

# Predictions
y_train_pred = classifier.model.predict(classifier.X_train_scaled)
y_test_pred = classifier.model.predict(classifier.X_test_scaled)

# Training accuracy
train_accuracy = accuracy_score(classifier.y_train, y_train_pred)
print(f"\nTraining Accuracy: {train_accuracy:.4f}")

# Test accuracy
test_accuracy = accuracy_score(classifier.y_test, y_test_pred)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Classification report
print("\n" + "-" * 60)
print("CLASSIFICATION REPORT (Test Set)")
print("-" * 60)
print(classification_report(classifier.y_test, y_test_pred))

# Confusion matrix
cm = confusion_matrix(classifier.y_test, y_test_pred)
print("\nConfusion Matrix:")
print(cm)

# Feature importance (coefficients)
print("\n" + "-" * 60)
print("FEATURE IMPORTANCE ANALYSIS")
print("-" * 60)

feature_names = ['beds', 'baths', 'sqft', 'price']
zip_codes = sorted(classifier.df['zip_code'].unique())
coefficients = classifier.model.coef_

print("\nModel Coefficients:")
for idx, zip_code in enumerate(zip_codes):
    print(f"\nZip Code {zip_code}:")
    for feat_idx, feature in enumerate(feature_names):
        print(f"{feature:10s}: {coefficients[idx][feat_idx]:8.4f}")

# Save cleaned data and results
classifier.df.to_csv('data/house_data_cleaned.csv', index=False)
print(f"\nCleaned data saved to: data/house_data_cleaned.csv")

# Save model results
results = {
    'model_type': 'Logistic Regression',
    'features': ['beds', 'baths', 'sqft', 'price'],
    'num_classes': len(classifier.df['zip_code'].unique()),
    'train_size': len(classifier.X_train),
    'test_size': len(classifier.X_test),
    'train_accuracy': train_accuracy,
    'test_accuracy': test_accuracy
}

with open('data/model_results.txt', 'w') as f:
    f.write("=" * 60 + "\n")
    f.write("LOGISTIC REGRESSION MODEL RESULTS\n")
    f.write("=" * 60 + "\n\n")

    for key, value in results.items():
        f.write(f"{key}: {value}\n")

    f.write("\n" + "-" * 60 + "\n")
    f.write("CLASSIFICATION REPORT\n")
    f.write("-" * 60 + "\n")
    f.write(classification_report(classifier.y_test, y_test_pred))

    f.write("\n" + "-" * 60 + "\n")
    f.write("CONFUSION MATRIX\n")
    f.write("-" * 60 + "\n")
    f.write(str(cm))

print(f"Model results saved to: data/model_results.txt")

# Save train/test split data for R visualizations
# Save predictions for visualization
results_df = pd.DataFrame({
    'actual': classifier.y_test,
    'predicted': y_test_pred
})
results_df.to_csv('data/test_predictions.csv', index=False)

# Save feature importance for visualization
coef_df = pd.DataFrame(
    coefficients,
    columns=feature_names,
    index=[f'zip_{z}' for z in zip_codes]
)
coef_df.to_csv('data/model_coefficients.csv')

# Save training data with predictions for analysis
train_results_df = pd.DataFrame({
    'actual': classifier.y_train,
    'predicted': y_train_pred
})
train_results_df.to_csv('data/train_predictions.csv', index=False)

print(f"Predictions saved to: data/test_predictions.csv")
print(f"Coefficients saved to: data/model_coefficients.csv")

print("\n" + "=" * 60)
print("PYTHON ANALYSIS COMPLETE")
print("=" * 60)
print(f"\nFinal Test Accuracy: {test_accuracy:.4f}")
print("\nGenerated files in data/:")
print("house_data.csv")
print("house_data_cleaned.csv")
print("model_results.txt")
print("test_predictions.csv")
print("train_predictions.csv")
print("model_coefficients.csv")
