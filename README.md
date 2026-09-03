# sentiment-Analysis

Classifying text as Positive, Negative, or Neutral using classic NLP and machine learning — no deep learning required.

Overview

This project builds an end-to-end sentiment classification pipeline:

Clean and preprocess raw text (lowercasing, URL/mention/hashtag removal, tokenization, stopword removal, lemmatization)
Convert text to numeric features using TF-IDF (unigrams + bigrams)
Train and compare two classifiers: Multinomial Naive Bayes and Logistic Regression
Evaluate with accuracy, precision, recall, F1-score, and confusion matrices
Visualize class distribution and per-class vocabulary with word clouds
Perform error analysis on misclassified examples
Dataset

The notebook is designed to work with the Twitter US Airline Sentiment dataset from Kaggle. If Tweets.csv is not found in the working directory, the notebook automatically falls back to generating a balanced synthetic dataset (1,800 samples, 600 per class) so it runs end-to-end with no setup required.

Note: The results and images included in this repo were generated using the synthetic fallback dataset, which uses a small set of template sentences repeated many times. This makes the classification task artificially easy (hence the perfect scores below). Results on the real Kaggle dataset will be more realistic and lower.

Pipeline
1. Text Preprocessing
Lowercase all text
Strip URLs, @mentions, and #hashtags
Remove non-alphabetic characters
Tokenize with NLTK's word_tokenize
Remove stopwords and short tokens
Lemmatize with WordNet
2. Feature Extraction

TfidfVectorizer with:

max_features=10000
ngram_range=(1, 2) — captures short phrases like "not good"
min_df=2

TF-IDF is fit on the training set only to avoid data leakage.

3. Models
Model	Notes
Multinomial Naive Bayes (alpha=0.1)	Fast probabilistic baseline, assumes feature independence
Logistic Regression (max_iter=1000, C=1.0)	Linear model that learns weighted feature combinations

Data is split 80/20 (train/test) with stratification, random_state=42.

Results

On the synthetic dataset, both models achieve perfect classification:

Model	Accuracy	Precision	Recall	F1-Score
Naive Bayes	1.000	1.000	1.000	1.000
Logistic Regression	1.000	1.000	1.000	1.000

Class Distribution

Vocabulary by Sentiment Class

Error Analysis

Common causes of misclassification in sentiment analysis, discussed in the notebook:

Negation — e.g. "not bad" is positive, but bag-of-words features skew negative from the word "bad"
Sarcasm/irony — e.g. "Oh great, delayed again" contains "great" but is negative
Mixed sentiment — a sentence praising one aspect while criticizing another
Domain-specific slang — underrepresented in training data
Short/vague text — minimal signal for the model to use
Real-World Applications
Application	How it helps
Customer support triage	Auto-flag negative tickets for priority handling
Brand monitoring	Track public opinion after launches or news events
Review aggregation	Summarize thousands of reviews into sentiment scores
Market research	Measure reactions to pricing or campaign changes
Political analysis	Gauge public sentiment toward policies in real time
Tech Stack
Python
pandas, numpy
scikit-learn
NLTK
matplotlib, seaborn
WordCloud
Getting Started
Installation
bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn wordcloud
Usage
(Optional) Download Tweets.csv from the Kaggle dataset and place it in the project root for real-world results.
Open and run sentiment_analysis.ipynb top to bottom.
Generated plots are saved automatically as .png files in the project directory.
Next Steps
Swap in transformer-based models (BERT, RoBERTa) for better handling of negation and sarcasm
Tune hyperparameters with GridSearchCV
Apply SMOTE if using a real, class-imbalanced dataset
Try ensemble/stacking of Naive Bayes and Logistic Regression
