# README Descriptions for Codsoft Machine Learning Projects

## Project 1: Bank Customer Churn Prediction

### Overview
This project aims to predict the likelihood of bank customers churning (i.e., leaving the bank) using a machine learning model. The RandomForest Classifier is utilized for its robustness and ability to handle a large dataset with a complex mix of numerical and categorical variables. This model predicts customer churn based on demographics and transactional behavior patterns.

### Features
- **Data Preprocessing**: Involves encoding categorical variables and preparing the dataset by removing irrelevant columns.
- **Model Training**: Utilizes a RandomForest Classifier, which is known for its high accuracy and capability to rank the importance of various features.
- **Model Evaluation**: Includes accuracy measurement, a detailed classification report, and ROC curve analysis to assess the model's predictive power and reliability.

### How to Run
1. Load the CSV data file and preprocess it by encoding categorical features and splitting the dataset into features (X) and the target variable (y).
2. Split the dataset into training and testing datasets.
3. Initialize and train the RandomForest classifier.
4. Evaluate the model's performance using accuracy metrics and visualize it using an ROC curve.

### Results
The project provides detailed metrics such as accuracy, a classification report, and an ROC curve plot, which helps in understanding the model's effectiveness in predicting customer churn.

---

## Project 2: Movie Genre Classification

### Overview
The Movie Genre Classification project categorizes movies into genres based on textual descriptions. It leverages Natural Language Processing (NLP) methods and a Naive Bayes classifier to analyze and predict movie genres, making it a useful tool for content categorization in digital media platforms.

### Features
- **Text Preprocessing**: Converts text data into a clean, normalized format.
- **Feature Extraction**: Uses TF-IDF vectorization to transform text data into a feature matrix that highlights important textual features for classification.
- **Model Training and Evaluation**: Employs a Naive Bayes classifier, evaluates its performance on a validation set, and provides predictions on a test set.

### How to Run
1. Prepare your dataset containing movie titles, genres, and descriptions.
2. Apply text preprocessing to clean and standardize descriptions.
3. Use TF-IDF vectorization for feature extraction.
4. Train the Naive Bayes classifier and evaluate it on the validation set.
5. Predict movie genres for the test dataset and save the predictions.

### Results
Outputs a validation set accuracy score, a classification report detailing performance metrics, and a CSV file with genre predictions for new movies.

---

## Project 3: Spam SMS Detection

### Overview
The Spam SMS Detection project is an application designed to filter spam from legitimate (ham) SMS messages. It uses a combination of NLP techniques and a multinomial Naive Bayes classifier to efficiently categorize incoming messages.

### Features
- **Data Preprocessing**: Filters and prepares SMS data for processing.
- **Text Normalization**: Includes converting text to lowercase, tokenization, removing stopwords, and stemming.
- **Feature Extraction**: Implements TF-IDF vectorization to convert processed text into numerical data suitable for machine learning.
- **Model Training**: Trains a multinomial Naive Bayes classifier capable of distinguishing between spam and ham messages.
- **Performance Evaluation**: Measures the model's performance using accuracy and precision scores.

### How to Run
1. Load the SMS data from a CSV file.
2. Perform data cleaning and text preprocessing to prepare the data.
3. Extract features using TF-IDF vectorization.
4. Split the data into training and testing sets.
5. Train the Naive Bayes model and evaluate its effectiveness.

### Results
The project includes outputs such as the accuracy and precision of the model, indicating how well it can identify spam messages.
