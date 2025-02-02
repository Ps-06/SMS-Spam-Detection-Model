# SMS Spam Detection Model Using NLP

This repository contains a spam detection model built using Natural Language Processing (NLP) techniques. The model uses three different Naive Bayes classifiers: GaussianNB, BernoulliNB, and MultinomialNB. This project aims to classify SMS messages as either spam or non-spam based on their content.

## Model Overview

The model is designed to process text data and classify it into two categories:
1. **Spam**: Unwanted promotional messages.
2. **Non-Spam**: Valid messages such as friend requests, service announcements, etc.

### Technologies Used
- **Python**: The primary programming language used for building the model.
- **Scikit-learn**: Library used for implementing machine learning algorithms and data preprocessing tasks.
- **Natural Language Processing (NLP)**: Techniques to process and analyze text data.

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
```

### 2. Text Processing
The preprocessing steps include:
- **Lowercasing**: Convert all text to lowercase.
- **Tokenization**: Split sentences into individual words or tokens.
- **Stopword Removal**: Remove common English stopwords like "a", "the", etc.

```bash
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize and remove special characters
    tokens = []
    for token in re.findall(r'\w+[\w\'-]*\w+', text):
        if token not in stopwords.words('english'):
            tokens.append(token)
    
    return ' '.join(tokens)

```

### 3. Feature Extraction
Convert the preprocessed text into numerical features using TF-IDF.
```bash
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TF-IDF vectorizer
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(preprocessed_texts)

# Convert to dense array for demonstration
X = X.toarray()

```

### 4. Model Training and Evaluation
Train each model and evaluate its performance.

```bash
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Train each model
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_gnb = gnb.predict(X_test)

bernoulli = BernoulliNB()
bernoulli.fit(X_train, y_train)
y_pred_bernoulli = bernoulli.predict(X_test)

multinomial = MultinomialNB()
multinomial.fit(X_train, y_train)
y_pred_multinomial = multinomial.predict(X_test)

# Evaluate models
def evaluate(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc = roc_auc_score(y_true, y_pred)
    return {'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F-1 Score': f1,
            'ROC AUC': roc}

# Evaluate each model
gnb_scores = evaluate(y_test, y_pred_gnb)
bernoulli_scores = evaluate(y_test, y_pred_bernoulli)
multinomial_scores = evaluate(y_test, y_pred_multinomial)

print("GaussianNB Scores:", gnb_scores)
print("BernoulliNB Scores:", bernoulli_scores)
print("MultinomialNB Scores:", multinomial_scores)

```

### 5. Parameter Tuning
Optimize the models by tuning hyperparameters.

```bash
# Example of hyperparameter grid for GaussianNB
gnb_param = {'var_smoothing': [1e-9, 1e-8, 1e-7]}

# Grid search for GaussianNB
gnb_gscv = GridSearchCV(gnb, gnb_param, cv=5)
gnb_gscv.fit(X_train, y_train)

print("Best Parameters for GaussianNB:", gnb_gscv.best_params_)

```

### 6. Model Deployment
Deploy the trained model to classify new SMS messages.

```bash
def predict_spam(text):
    # Preprocess text
    preprocessed = preprocess_text(text)
    
    # Convert to TF-IDF features
    tfidf_features = tfidf.transform([preprocessed])
    
    # Make predictions using best-performing model
    prediction = multinomial_gscv.predict(tfidf_features)[0]
    
    return "spam" if prediction == 1 else "non_spam"

# Example usage
message = "Free entry in this week's lotto draw. 21st January 2024. T&Cs apply."
print("Prediction:", predict_spam(message))

```
### 7. Evaluation Metrics
Key metrics to evaluate model performance include:
- **Accuracy**: Percentage of correct predictions.
- **Precision**: Ratio of correctly predicted positive observations to the total predicted positives.
- **Recall**: Ratio of correctly predicted positive observations to the total actual positives.
- **F-1 Score**: Harmonic mean of precision and recall.
- **ROC AUC**: Area under the receiver operating characteristic curve.

### 8. Case Study
An analysis of a real-world dataset shows that the model achieves `[insert accuracy]%` with `[insert precision]%` precision, demonstrating its effectiveness in distinguishing spam from non-spam messages.

# Deployment Instructions
To deploy this model:

1. Install Dependencies:
   ```bash
   pip install python-dotenv scikit-learn pandas numpy nltk
   ```
2. Run Training Script:
   ```bash
   python train_model.py
   ```
3. Use the Model for Predictions:
   - Load the trained model using pickle.
   - Preprocess new messages and make predictions.
4. Monitor Performance:
   - Continuously evaluate the model's performance as more data comes in.
   - Retrain the model periodically with fresh data to maintain accuracy.
  
# Considerations
- 
