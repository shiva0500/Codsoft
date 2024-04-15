import numpy as np
import pandas as pd
import re
import nltk
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Downloading necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load training and testing data from text files
train_dataset = pd.read_csv(r'Movie_Genre_classifier\Genre Classification Dataset\train_data.txt', 
                            sep=':::', names=['Title', 'Genre', 'Description'], engine='python')
test_dataset = pd.read_csv(r'Movie_Genre_classifier\Genre Classification Dataset\test_data.txt', 
                           sep=':::', names=['Title', 'Genre', 'Description'], engine='python')

# Function to clean text data
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'@\S+', '', text)  # Remove mentions
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'pic.\S+', '', text)  # Remove picture URLs
    text = re.sub(r"[^a-zA-Z+']", ' ', text)  # Keep only letters and apostrophes
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text + ' ')  # Remove single letters
    text = "".join([char for char in text if char not in string.punctuation])  # Remove punctuation
    words = nltk.word_tokenize(text)  # Tokenize text
    stopwords_set = nltk.corpus.stopwords.words('english')  # Get stopwords
    text = " ".join([word for word in words if word not in stopwords_set and len(word) > 2])  # Remove stopwords
    text = re.sub(r"\s[\s]+", " ", text).strip()  # Remove extra spaces
    return text

# Apply text cleaning to training and testing data
train_dataset['Text_cleaning'] = train_dataset['Description'].apply(clean_text)
test_dataset['Text_cleaning'] = test_dataset['Description'].apply(clean_text)

# Vectorization of text using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X_train = tfidf_vectorizer.fit_transform(train_dataset['Text_cleaning'])
X_test = tfidf_vectorizer.transform(test_dataset['Text_cleaning'])

# Prepare labels and split data for training and validation
y = train_dataset['Genre']
X_train, X_val, y_train, y_val = train_test_split(X_train, y, test_size=0.2, random_state=42)

# Classifier instantiation and fitting
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Prediction and evaluation
y_pred = classifier.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print("Validation Accuracy:", accuracy)
print(classification_report(y_val, y_pred))

# Predicting genres for test dataset and saving the results
X_test_predictions = classifier.predict(X_test)
test_dataset['Predicted_Genre'] = X_test_predictions
test_dataset.to_csv(r'Movie_Genre_classifier\predicted_genres.csv', index=False)
print(test_dataset)
