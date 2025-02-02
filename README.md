# SMS-Spam-Detection-Model
SMS Spam Detection Model using NLP
This README file provides an overview of an SMS spam detection model developed using Natural Language Processing (NLP) techniques. The model aims to classify SMS messages as either spam or non-spam based on their content.

Project Overview
The project involves creating a machine learning model to detect spam messages from legitimate ones. The model uses NLP techniques to analyze text data and categorize messages into two classes: spam or non-spam (ham). The following classifiers are used in this project:

Gaussian Naive Bayes (GNB)
Bernoulli Naive Bayes (BNB)
Multinomial Naive Bayes (MNB)
Features and Preprocessing
The model processes SMS messages through the following steps:

Text Cleaning: Removes special characters, numbers, and punctuations.
Tokenization: Splits the text into individual words or tokens.
Stopword Removal: Removes common English stopwords like "and," "the," etc.
TF-IDF Transformation: Converts the tokenized text into TF-IDF weighted features for better representation of important words.
Classifiers
The model employs three different Naive Bayes classifiers to handle different types of data distributions:

Gaussian Naive Bayes (GNB): Suitable for continuous data, which is transformed using TF-IDF.
Bernoulli Naive Bayes (BNB): Handles binary/boolean features, useful when the text is represented as word occurrences.
Multinomial Naive Bayes (MNB): Best suited for count-based data, where message features are based on word frequencies.
Evaluation Metrics
The model's performance is evaluated using the following metrics:

Accuracy: Overall correctness of predictions.
Precision: Ratio of correctly predicted spam messages to all predicted spam messages.
Recall: Ratio of correctly predicted spam messages to all actual spam messages.
F1-Score: Harmonic mean of precision and recall, providing a balanced measure of both.
Confusion Matrix: A table that summarizes the performance of the classification model.
Installation
To install the project's dependencies, you can use pip:

pip install scikit-learn==0.24.1 numpy pandas NLTK
Model Training Process
The training process involves the following steps:

Load the dataset and split it into training and testing sets.
Apply preprocessing techniques to both train and test datasets.
Initialize each classifier with default hyperparameters.
Fit each model on the training data.
Predict labels for the test data using each model.
Evaluate performance metrics for each model.
Usage
To use the model, execute:

python spam-detection.py --model [gnb|bnb|mnb] --data [path/to/data]
Where:

[model] specifies which classifier to use (gnb/bnb/mnb).
[path/to/data] is the directory containing the training data.
Example Output
A sample output of model performance might look like this:

Accuracy: 0.98
Precision for spam: 0.97
Recall for spam: 0.96
F1-Score for spam: 0.96
Confusion Matrix:
[[5423    12]
 [  12   98]]
Contributions
Contributions are welcome! If you have suggestions or improvements, please submit an issue or pull request on the project's GitHub repository.

License
This code is distributed under the MIT License. For more details, see theLICENSE file in the repository.
