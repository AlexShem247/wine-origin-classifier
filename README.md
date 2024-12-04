# Wine Origin Classifier

This repository contains code and data for classifying the country of origin of wine and using machine learning models to predict wine it based on different features. It includes data processing, machine learning, and feature extraction using LLMs (Large Language Models). The project utilises various data sources, such as wine descriptions and quality scores, to build predictive models.

## Directory Structure

```plaintext
ArtificialLabs/
├── data/                           # Data files
│   ├── wine_quality_10.csv         # Small dataset of wine quality
│   ├── wine_quality_1000.csv       # Full dataset of wine quality
│   └── wine_quality_1000.xlsx      # Excel file version
│
├── images/                         # Folder containing images for the report
│   └── decision_tree.png           # Image of a decision tree visualisation
│
├── output/                         # Output files produced by the LLM
│   ├── predictions_1000.csv        # Predictions for 1000 wines without any training
│   ├── wine_features_1000.csv      # Features extracted from the wine description
│   ├── wine_quality_features+-1000.csv  # Wine dataset with LLM filtered BoW
│   └── wine_quality_more_features_500.csv  # Wine dataset with additional features from desc.
│
├── report/                         # Report folder
│   ├── report.md                   # Markdown report of the project
│   └── report.pdf                  # Final report in PDF format
│
├── src/                            # Source code for model and feature extraction
│   ├── LLMClassifier.py            # Model for untrained wine quality classification
│   ├── LLMClient.py                # Client to interact with the LLM API
│   ├── LLMFeatureExtractor.py      # Extract features using LLM
│   └── main.py                     # Main script to run the project
│
├── .gitignore                      # Git ignore file to exclude unnecessary files
└── README.md                       # This README file
```

## Project Overview

This project uses various datasets to explore the relationship between different wine features and their quality. The goal is to predict the origin, type, and overall quality of wines using machine learning models. The project leverages **Bag of Words** and **Large Language Models (LLM)** for feature extraction from wine descriptions, in combination with traditional wine attributes like points, price, and acidity.

### Key Features:
- **Wine Quality Prediction**: Predict wine quality based on its features and descriptions.
- **Feature Extraction**: Utilise both traditional features (e.g., price, acidity) and LLM-based feature extraction from wine descriptions.
- **Data Visualisation**: Generate decision trees to visualise model decisions.

## Requirements

To run this project, you'll need the following Python libraries:

- `graphviz==0.20.3`
- `pandas==2.2.3`
- `Requests==2.32.3`
- `scikit_learn==1.5.2`

## How to Run

To run the classifier, navigate to the `src` directory and run:

```bash
python main.py
```

To run the `LLMClassifier` or `LLMFeatureExtractor`, use the following command, replacing `<API_KEY>` with your actual API key:

```bash
python LLMClassifier.py <API_KEY>
```

or

```bash
python LLMFeatureExtractor.py <API_KEY>
```