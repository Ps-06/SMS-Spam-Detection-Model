# SMS Spam Detection Model Using NLP

This repository contains a spam detection model built using Natural Language Processing (NLP) techniques. The model uses three different Naive Bayes classifiers: GaussianNB, BernoulliNB, and MultinomialNB. This project aims to classify SMS messages as either spam or non-spam based on their content.

## Model Overview

The model is designed to process text data and classify it into two categories:
1. **Spam**: Unwanted promotional messages.
2. **Non-Spam**: Valid messages such as friend requests, service announcements, etc.

### Technologies Used
- Python: The primary programming language used for building the model.
- Scikit-learn: Library used for implementing machine learning algorithms and data preprocessing tasks.
- Natural Language Processing (NLP): Techniques to process and analyze text data.

## How It Works

The model follows these steps to classify messages:
1. **Data Collection**: Gather a dataset of SMS messages labeled as spam or non-spam.
2. **Text Preprocessing**: Clean the text by removing irrelevant information, handling case sensitivity, and converting text into numerical features.
3. **Feature Extraction**: Convert processed text into numerical vectors using techniques like Term Frequency-Inverse Document Frequency (TF-IDF).
4. **Model Training**: Train three Naive Bayes classifiers on the labeled dataset.
5. **Model Evaluation**: Evaluate the models' performance using various metrics to select the best-performing model.

## Implementation Steps

### 1. Data Collection
Collect a dataset of SMS messages with labels (`spam` or `non_spam`). A sample dataset can be found [here](https://bit.ly/3s8E7ei).

```bash
# Download the dataset
wget https://bit.ly/3s8E7ei
