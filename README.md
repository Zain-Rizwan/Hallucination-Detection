# Hallucination Detection Classifier Using N-Grams

This repository contains the implementation of a **binary hallucination detection classifier** that identifies whether a given text (summary) is factual or contains hallucinations. The model is trained on the **XSum Hallucination Dataset** and uses logistic regression without libraries for classification.

## Features
- **Logistic Regression**: Implements logistic regression from scratch for binary classification.
- **Data Preprocessing**: Text cleaning, tokenization, and feature extraction.
- **Evaluation Metrics**: Accuracy, precision, recall, F1-score, and k-fold cross-validation.
- **Error Analysis**: In-depth analysis of model predictions and misclassifications.
- **Visualizations**: Includes plots for cross-validation and error metrics.

## Dataset
The model is trained and tested on the [XSum Hallucination Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews). This dataset contains summaries labeled as factual or hallucinated. (Ensure you have the dataset in the appropriate format before training.)

## How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/Zain-Rizwan/Hallucination-Detection-using-N-Gram.git
