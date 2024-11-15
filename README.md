# French Name Classifier

## Project Overview

The French Name Classifier project was developed for Bleu Blanc Rouge VC with the aim of identifying French founders at early stages of funding, enabling strategic partnerships and investments. This machine learning solution predicts if a person is French based solely on their first name, utilizing various classification models to maximize accuracy and effectiveness. The project is part of the Machine Learning for Business II course and adheres to best practices in data science and machine learning.

## Table of Contents

- [French Name Classifier](#french-name-classifier)
  - [Project Overview](#project-overview)
  - [Table of Contents](#table-of-contents)
  - [Business Case](#business-case)
  - [Dataset](#dataset)
    - [Data Sources](#data-sources)
    - [Data Preprocessing](#data-preprocessing)
  - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
  - [Models](#models)
    - [Baseline Model](#baseline-model)
    - [Model Comparison](#model-comparison)
    - [Model Fine-Tuning](#model-fine-tuning)
  - [Metrics](#metrics)
  - [Application and Deployment](#application-and-deployment)
    - [Local Deployment](#local-deployment)
      - [On Windows (using `waitress`)](#on-windows-using-waitress)
      - [On Unix-based Systems (using `gunicorn`)](#on-unix-based-systems-using-gunicorn)
    - [Deployed Website](#deployed-website)

---

## Business Case

Bleu Blanc Rouge VC's mission is to identify and invest in promising French startup founders. This project focuses on:

1. Identifying potential French founders based on their first names.
2. Utilizing machine learning to automate the classification process, improving lead qualification.
3. Providing a scalable, deployable model through a web application and API.

## Dataset

### Data Sources

1. **Scraper Results**: Data collected through web scraping, including names, URLs, and related data.
2. **INSEE First Name Report**: A dataset containing counts of names per department per year since 1900.

### Data Preprocessing

- Extracted relevant columns: `first_name` and `is_french`.
- Removed rows with dates prior to 1960.
- Scaled frequencies to account for count imbalances.
- Merged data from multiple sources and cleaned names by removing special characters, emojis, and handling joint names.
- Resulting dataset: 20,000+ rows with two columns: `first_name` and `is_french`.

## Exploratory Data Analysis (EDA)

Detailed EDA can be found in the `eda.ipynb` file. Key insights include:

- Distribution of French vs. non-French names.
- Data cleaning and preprocessing steps.
- Feature exploration and selection strategies.

## Models

### Baseline Model

**Logistic Regression**:

- Used as the initial model to establish a performance baseline.
- Features: TF-IDF vectorization of first names.
- Metrics (on test set): Precision, Recall, F1 Score, Accuracy, and ROC AUC.

### Model Comparison

Three models were evaluated:

1. **Logistic Regression**
2. **Random Forest** (Ensemble method with multiple decision trees)
3. **Naive Bayes** (Assumes independence between features)

Results:

- **Random Forest** performed the best overall with the highest F1 score, demonstrating a good balance between precision and recall.
- **Logistic Regression** was used as the baseline and provided good interpretability.
- **Naive Bayes** was fast but less accurate.

### Model Fine-Tuning

**Random Forest Fine-Tuning**:

- Hyperparameters tuned using Random Search Cross-Validation.
- Parameter grid included variations for `n_estimators`, `max_depth`, `min_samples_split`, and `min_samples_leaf`.

## Metrics

Metrics used for evaluating models include:

- **Accuracy**: Proportion of correct predictions.
- **Precision**: Ratio of correctly predicted positive observations to total predicted positives.
- **Recall**: Ratio of correctly predicted positive observations to all observations in the actual class.
- **F1 Score**: Harmonic mean of Precision and Recall.
- **ROC AUC**: Area under the Receiver Operating Characteristic curve.

## Application and Deployment

### Local Deployment

To run the Flask application locally, use the following instructions based on your operating system:

#### On Windows (using `waitress`)

1. Install `waitress`:

   ```bash
   pip install waitress
   ```
  
2. Run the application:

   ```bash
   waitress-serve --port=5000 app:app
   ```

This will serve your application on `http://localhost:5000`.

#### On Unix-based Systems (using `gunicorn`)

1. Install `gunicorn`:

   ```bash
   pip install gunicorn
   ```

2. Run the application:

   ```bash
   gunicorn app:app
   ```

### Deployed Website

The French Name Classifier web application is deployed and accessible at:
**[https://french-classifier-by-first-name.onrender.com/](https://french-classifier-by-first-name.onrender.com/)**
