# Imports
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score
from sklearn.preprocessing import LabelEncoder
import string

# Data loading
dataset = pd.read_csv(r'Spam_Sms_Detection\messages.csv', encoding="ISO-8859-1")

# Data preprocessing
dataset.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
dataset.rename(columns={'v1': 'Target', 'v2': 'Text'}, inplace=True)
dataset.drop_duplicates(inplace=True)

encoder = LabelEncoder()
dataset['Target'] = encoder.fit_transform(dataset['Target'])

# Split the DataFrame into two based on 'Target' column
ham_dataset = dataset[dataset['Target'] == 0]  # where Target is 0, meaning 'ham'
spam_dataset = dataset[dataset['Target'] == 1]  # where Target is 1, meaning 'spam'

# Save DataFrames to separate CSV files
ham_dataset.to_csv(r'Spam_Sms_Detection\Classified_ham.csv', index=False)
spam_dataset.to_csv(r'Spam_Sms_Detection\Classified_spam.csv', index=False)

print("Files saved: 'ham.csv' and 'spam.csv'")



# Define text processing function
def process_text(text):
    # Convert to lower case
    text = text.lower()
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords and punctuation
    tokens = [word for word in tokens if word not in stopwords.words('english') and word not in string.punctuation]
    # Stemming
    ps = PorterStemmer()
    tokens = [ps.stem(word) for word in tokens]
    return " ".join(tokens)

# Apply text processing
dataset['Transformed_text'] = dataset['Text'].apply(process_text)


# Feature Extraction
tfidataset = TfidfVectorizer(max_features=3000)
X = tfidataset.fit_transform(dataset['Transformed_text'])
y = dataset['Target'].values

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X.toarray(), y, test_size=0.2, random_state=2)

# Model training
mnb = MultinomialNB()
mnb.fit(X_train, y_train)
# Predictions and evaluations
y_pred = mnb.predict(X_test)
print(f"Accuracy Score: {accuracy_score(y_test, y_pred)}")
print(f"Precision Score: {precision_score(y_test, y_pred)}")
