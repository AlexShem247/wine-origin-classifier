# Wine Origin Classifier: Predicting Country of Origin of Wines using ML and LLMs

**Name**: Alexander Shemaly

**Date**: 3rd December 2024

## Table of Contents
1. [Introduction](#1-introduction)
2. [Data Overview](#2-data-overview)
   1. [Data Imbalance](#21-data-imbalance)
   2. [Model Baseline](#22-model-baseline)
   3. [Wine Descriptions](#23-wine-descriptions)
3. [Approach and Methodology](#3-approach-and-methodology)
   1. [Predictive Modeling Using Quantitative Data](#31-predictive-modeling-using-quantitative-data)
   2. [NLP for Textual Feature Extraction](#32-nlp-for-textual-feature-extraction)
   3. [LLM Integration](#33-llm-integration)
4. [Challenges and Considerations](#4-challenges-and-considerations)
5. [Conclusion](#5-conclusion)

## 1. Introduction

The purpose of this project was to build a classification model that predicts
the country of origin of wines based on a dataset containing wine reviews.

This report outlines the approach taken to preprocess the data, handle the challenges posed by imbalanced data and categorical features, and evaluate various models, including the  use of Large Language Models (LLMs).

## 2. Data Overview

The dataset includes five columns: **Country**, **Description**, **Points**, **Price**, and **Variety**:

- **Country**: The target variable, representing the country of origin of the wine. It can be one of four categories: **US**, **Spain**, **France**, or **Italy**.
- **Description**: A plain-text description of the wine, typically containing 30-40 words. This field provides details about the wine’s characteristics, flavors, and other sensory attributes.
- **Points**: An integer value representing the average review score of the wine on a scale of 1-100.
- **Price**: An integer value indicating the cost of a bottle of wine.
- **Variety**: A string representing the type of grape used to make the wine. This is a categorical feature with several possible values.

### 2.1 Data Imbalance

Upon examining the distribution of the target variable, **Country**, it is clear that the dataset is highly imbalanced, as shown below:

- **US**: 622 entries
- **Italy**: 174 entries
- **France**: 133 entries
- **Spain**: 71 entries

This indicates that the model will be more likely to predict the **US** correctly simply due to its higher representation. The imbalance in the dataset poses a challenge for the underrepresented classes, particularly **Spain**, **France**, and **Italy**, which may be under-predicted by the model.

To account for this imbalance, we will use weighted evaluation metrics, such as the weighted average F1-score, which considers the proportion of each class in the dataset. This will help ensure that the model's performance on the less frequent classes is properly reflected.

### 2.2 Model Baseline

With four possible countries to predict, a model that predicts randomly would have an expected accuracy of 25%.

However, since the **US** is the most frequent class (by a significant margin), a model that always predicts **US** would have an accuracy of approximately 62%. This reflects the imbalance in the dataset and highlights the risk of overfitting to the majority class, where **US** is predicted most of the time.

Given this, a baseline accuracy of **62%** is a reasonable benchmark. The model we develop should aim to exceed this baseline by improving its predictions for the underrepresented classes, particularly **Spain**, **France**, and **Italy**.

### 2.3 Wine Descriptions

The **Description** column in the dataset provides detailed insights into the sensory attributes of each wine. These attributes can offer valuable clues for predicting the **Country** of origin, as different regions tend to produce wines with distinct profiles.

The following wine features could help identify the country of origin of the wine:

- **Flavor Profiles**: Terms like *berry*, *pepper*, and *mushroom* indicate fruity, spicy, or earthy characteristics, suggesting the wine's flavor profile.

- **Texture and Body**: Descriptors such as *light-bodied*, *full-bodied*, and *tannic* reveal the wine's weight and texture.

- **Acidity and Sweetness**: Words like *high acidity* and *dry* convey the wine’s freshness and sugar content.

- **Aging and Ageability**: Phrases like *ready to drink* or *needs aging* describe the wine’s maturity and aging potential.

- **Terroir and Region**: Terms such as *coastal* or *volcanic* point to the influence of the wine's origin and terroir.


By analysing these features, we can identify patterns that might correlate with specific countries of origin. For example, **earthy** and **floral** flavors may point to **France** or **Italy**, while **fruity** and **spicy** notes could suggest wines from the **US** or **Spain**.

These features can be extracted using the **Large Language Model (LLM)** techniques to enhance the model and improve prediction accuracy.



## 3. Approach and Methodology

To predict the **country of origin** of wine, I will combine machine learning and natural language processing (NLP) techniques.

For the first part, I will begin by creating a baseline model using only the existing features in the dataset, such as **Points**, **Price**, and **Variety**, while ignoring the **description** field. This will help me establish a starting point and assess the contribution of these features to the prediction of the **country**.

For the second part, I will enhance the model by applying **NLP techniques** to the **description** field. This will include methods like **bag-of-words** and **TF-IDF vectorisation**, which will allow the model to capture important textual features. 


Finally, I will integrate **Large Language Models (LLM)** to extract relevant features and guide the text vectorisation process, enhancing model performance by reducing noise and focusing on contextually important features.

### 3.1 Predictive Modeling Using Quantitative Data

In this phase, I will build a predictive model using the existing quantitative features: **Points**, **Price**, and **Variety**. This will establish a baseline model to predict the **country of origin** of the wine and assess what can be achieved with the current data.

- **Points**: While representing the wine’s average review score, this feature may not strongly predict the country of origin, as wines from different regions can receive similar ratings.
- **Price**: Although it provides insight into the wine’s cost, price can vary widely across regions and may not effectively differentiate between countries.
- **Variety**: This feature is likely to be a strong predictor, as certain grape varieties are strongly associated with specific regions. However, with 105 categories, **one-hot encoding** will be used to transform **Variety** into binary columns.

#### 3.1 Choice of Model

For modeling, I will use a **Random Forest**, an ensemble method that combines multiple **decision trees** to improve accuracy. Each tree is trained on a random subset of data and features. A **decision tree** splits data based on a feature that provides the best **information gain**, which helps classify the data, in this case, the wine's country of origin. While a single decision tree can overfit the data, **Random Forest** reduces this by averaging predictions from many trees, making it more robust and accurate.

Although a **neural network** could also be used, it may overtrain the data, especially with the current feature set. Neural networks are powerful but can become too complex for this task, requiring more training time and computational resources. In contrast, Random Forest provides a simpler and more interpretable solution. The decision trees produced by Random Forest are valuable for understanding the logic behind predictions, which is particularly useful in a business context.

#### 3.2 Initial Implementation

foo

#### 3.2 Results

foo

Insert picture of decision tree here.

#### 3.3 Improvement using Hyperparameter Tuning with Cross-Validation

foo

#### 3.4 Results

foo

Insert best parameters here.

#### 3.5 Summary

foo


### 3.2 NLP for Textual Feature Extraction

### 3.3 LLM Integration