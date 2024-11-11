import os
import pickle
import re

import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

app = Flask(__name__)

# Load pre-trained model and vectorizer (assumes you have saved them previously)
MODEL_PATH = "model/model.pkl"
VECTORIZER_PATH = "model/vectorizer.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    raise FileNotFoundError(
        f"Model and vectorizer files must be saved in '{MODEL_PATH}' and '{VECTORIZER_PATH}'."
    )

with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)

with open(VECTORIZER_PATH, "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)


# Define text cleaning function
def clean_text(text):
    # Remove anything between double quotes
    text = re.sub(r"\".*?\"", "", text)
    # Remove anything between brackets
    text = re.sub(r"\(.*?\)", "", text)
    # Remove emojis
    emoji_pattern = re.compile(
        "["  # Emoji ranges
        "\U0001F600-\U0001F64F"  # Emoticons
        "\U0001F300-\U0001F5FF"  # Symbols & pictographs
        "\U0001F680-\U0001F6FF"  # Transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # Flags
        "\U00002500-\U00002BEF"  # Chinese characters
        "\U00002702-\U000027B0"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010FFFF"
        "\u2640-\u2642"
        "\u2600-\u2B55"
        "\u200d"
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\ufe0f"
        "\u3030"
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub(r"", text)
    # Remove non-alphanumeric characters except accents and apostrophes, and convert to lowercase
    text = re.sub(r"[^a-zA-ZÀ-ÖØ-öø-ÿ\' ]", "", text).lower()
    # Normalize whitespaces and keep only the first word
    text = " ".join(text.split()).split(" ")[0] if text else None
    return text


# Define function for cleaning data
def clean_input_data(df):
    df = df.copy()
    # Convert 'first_name' to string
    df["first_name"] = df["first_name"].astype(str)
    # Clean the 'first_name' column using the 'clean_text' function
    df["first_name"] = df["first_name"].apply(clean_text)
    # Drop rows with missing 'first_name' values after cleaning
    df = df.dropna(subset=["first_name"])
    return df


# Create a data transformation pipeline for cleaning
cleaning_pipeline = Pipeline(
    [("clean_data", FunctionTransformer(func=clean_input_data))]
)


@app.route("/", methods=["GET", "POST"])
def home():
    """
    Render the home page with a form to submit first names for prediction.
    """
    if request.method == "POST":
        original_first_name = request.form.get("first_name")
        if not original_first_name:
            return render_template("index.html", error="Please enter a first name.")

        # Create a DataFrame for input data and apply cleaning
        input_df = pd.DataFrame({"first_name": [original_first_name]})
        input_df = cleaning_pipeline.transform(input_df)

        if input_df.empty:
            return render_template(
                "index.html", error="Input 'first_name' is invalid after cleaning."
            )

        # Vectorize cleaned input
        X = vectorizer.transform(input_df["first_name"])

        # Make prediction
        prob = model.predict_proba(X)[0, 1]
        prediction = "French" if prob >= 0.5 else "Not French"

        # Format the cleaned name for display (convert to title case)
        formatted_name = input_df["first_name"].iloc[0].title()

        return render_template(
            "index.html",
            original_first_name=original_first_name,
            first_name=formatted_name,
            is_french_probability=round(prob, 4),
            prediction=prediction,
        )

    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    API endpoint for predicting if a given first name is likely French.
    """
    data = request.json
    if "first_name" not in data:
        return jsonify({"error": "Missing 'first_name' in request data"}), 400

    # Create a DataFrame for input data and apply cleaning
    input_df = pd.DataFrame({"first_name": [data["first_name"]]})
    input_df = cleaning_pipeline.transform(input_df)

    if input_df.empty:
        return jsonify({"error": "Input 'first_name' is invalid after cleaning."}), 400

    # Vectorize cleaned input
    X = vectorizer.transform(input_df["first_name"])

    # Make prediction
    prob = model.predict_proba(X)[0, 1]
    prediction = "French" if prob >= 0.5 else "Not French"

    response = {
        "first_name": data["first_name"],
        "is_french_probability": round(prob, 4),
        "prediction": prediction,
    }
    return jsonify(response)


# @app.route("/predict", methods=["POST"])
# def predict():
#     """
#     Predict if a given first name is likely French using the trained Random Forest model.

#     Request format (JSON):
#     {
#         "first_name": "Jean"
#     }

#     Response format (JSON):
#     {
#         "first_name": "Jean",
#         "is_french_probability": 0.85,
#         "prediction": "French" or "Not French"
#     }
#     """
#     data = request.json
#     if "first_name" not in data:
#         return jsonify({"error": "Missing 'first_name' in request data"}), 400

#     # Create a DataFrame for input data and apply cleaning
#     input_df = pd.DataFrame({"first_name": [data["first_name"]]})
#     input_df = cleaning_pipeline.transform(input_df)

#     # Check if cleaning resulted in an empty DataFrame
#     if input_df.empty:
#         return jsonify({"error": "Input 'first_name' is invalid after cleaning."}), 400

#     # Vectorize cleaned input
#     X = vectorizer.transform(input_df["first_name"])

#     # Make prediction
#     prob = model.predict_proba(X)[
#         0, 1
#     ]  # Probability for the positive class (is_french)
#     prediction = "French" if prob >= 0.5 else "Not French"

#     response = {
#         "first_name": input_df["first_name"].iloc[0],
#         "is_french_probability": round(prob, 4),
#         "prediction": prediction,
#     }
#     return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)
