import os
import pickle

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# Ensure output directory exists
os.makedirs("out", exist_ok=True)


# Load the dataset
def load_dataset(filepath):
    print("Loading dataset...")
    df = pd.read_csv(filepath)
    print(f"Dataset loaded with {len(df)} rows and {len(df.columns)} columns.")
    return df


# Train a model
def train_model(X_train, y_train):
    print("Training Random Forest model...")
    model = RandomForestClassifier(class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)
    print("Model training complete.")
    return model


# Compute and return model metrics
def compute_metrics(model, X_test, y_test):
    print("Computing metrics...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probability for the positive class

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC AUC": roc_auc_score(y_test, y_prob),
    }
    print("Metrics computed.")
    return metrics


# Write metrics to a file
def write_metrics_to_file(metrics, output_path="out/score.txt"):
    print(f"Writing metrics to {output_path}...")
    with open(output_path, "w") as f:
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
    print(f"Metrics written to {output_path}.")


def main():
    MLFLOW_TRACKING_URI = "http://127.0.0.1:8080/"
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Start MLFlow experiment
    mlflow.set_experiment("final_first_name_is_french_binary_classification")

    # Start MLflow run
    with mlflow.start_run():
        print("Starting MLflow run...")

        # Load and preprocess the data
        dataset_path = (
            "./data/final_first_name_dataset.csv"  # Change this path as needed
        )
        df = load_dataset(dataset_path)

        # Handle missing values
        print("Handling missing values...")
        df = df.dropna(subset=["first_name"])  # Drop rows where 'first_name' is NaN
        df["first_name"] = df["first_name"].astype(
            str
        )  # Ensure all entries are strings
        df["is_french"] = df["is_french"].astype(
            "boolean"
        )  # Ensure all entries are boolean
        print(f"Data cleaned. {len(df)} rows remaining.")

        # Feature extraction
        print("Extracting features with TF-IDF vectorization...")
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(df["first_name"])
        y = df["is_french"]
        print("Feature extraction complete.")

        # Train-test split
        print("Splitting data into training and testing sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print(f"Training set size: {X_train.shape[0]} samples")
        print(f"Testing set size: {X_test.shape[0]} samples")

        # Train the model
        model = train_model(X_train, y_train)

        # Log model parameters (example parameters to log)
        print("Logging model parameters...")
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("random_state", 42)

        # Compute metrics
        metrics = compute_metrics(model, X_test, y_test)

        # Log metrics to MLflow
        print("Logging metrics to MLflow...")
        mlflow.log_metrics(metrics)

        # Log the model
        print("Logging model to MLflow...")
        mlflow.sklearn.log_model(model, "random_forest_model")
        print("Model logged successfully.")

        # Write metrics to a file
        write_metrics_to_file(metrics)

        print("MLflow run complete.")

        # Save model
        print("Saving model...")
        with open("model/model.pkl", "wb") as model_file:
            pickle.dump(model, model_file)

        # Save vectorizer
        print("Saving vectorizer...")
        with open("model/vectorizer.pkl", "wb") as vectorizer_file:
            pickle.dump(vectorizer, vectorizer_file)

    mlflow.end_run()


if __name__ == "__main__":
    main()
