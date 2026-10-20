# Linguistic Features: Psychological Stress Prediction with Dreaddit

## Project Overview
The project investigates how linguistic features extracted from Reddit posts can predict binary stress-related labels. The dataset that I used is the Dreaddit dataset from Columbia University for stress analysis in social media (Turcan & McKeown, 2019).

The files are in the data/ directory:
* train-00000-of-00001.parquet
* test-00000-of-00001.parquet

## Requirements
Python 3.10 or a compatible version is necessary. The following packages are necessary:
* numpy
* pandas
* pyarrow
* matplotlib
* scipy
* scikit-learn

To install dependencies: `python3 -m pip install numpy pandas pyarrow matplotlib scipy scikit-learn`

# Running the Experiment
Run the experiment from the project root directory: `python3 stress_text_ml.py > outputs/results_final.txt 2>&1`

Ensure that the outputs/ directory exists before running the command.

# Limitations
This project focuses on a stress-related dataset label, not a clinical diagnosis.
