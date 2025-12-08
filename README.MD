Overview

This project performs end-to-end data preparation, feature engineering, dimensionality reduction, and multiple machine learning workflows including:

Classification

Regression

Clustering

Each stage is handled by a separate Python script for clarity.

1. Installation
Prerequisites

Python 3.10+

pip installed

Install Dependencies

All required packages are listed in requirements.txt.

pip install -r requirements.txt

2. Running the Project

The project is organized into four main scripts.
They should be executed in order, because later steps depend on files generated in earlier steps.

Step 1 — Data Cleaning & Feature Engineering
python phase1.py


What this script does:

Loads and cleans the dataset

Performs feature engineering

Applies dimensionality reduction

Saves the processed dataset for downstream models

This step must be run before any of the model scripts.

Step 2 — Classification Models
python classification.py


Notes:

This script trains and evaluates multiple classification algorithms.

Most models are commented out to avoid extremely long runtimes.

You may uncomment specific sections if you want to re-run individual model experiments.

Step 3 — Regression Model
python regression.py


What this script does:

Runs the linear regression workflow

Trains, evaluates, and outputs performance metrics

Step 4 — Clustering Models
python clustering.py


What this script does:

Runs unsupervised clustering experiments

Includes evaluation metrics and optional visualizations

3. Output Files

Depending on the script, the project may generate:

cleaned datasets

engineered feature sets

model metrics

plots (ROC curves, confusion matrices, heatmaps, etc.)

clustering visualizations

All outputs are saved automatically to the designated results directory inside the project.

4. Reproducibility

To ensure consistent results:

Always run the scripts in order

Make sure your requirements.txt environment is installed cleanly

Use the same dataset(s) included in the project root

5. Contact

If you have questions about running the code or interpreting results, feel free to reach out.