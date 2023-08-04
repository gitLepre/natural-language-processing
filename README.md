# Real or Not? NLP with Disaster Tweets

This repository contains a Python notebook for a Natural Language Processing (NLP) project called "Real or Not? NLP with Disaster Tweets." The goal of this project is to classify tweets as either real disaster tweets or non-disaster tweets.

## Project Overview

The notebook uses various NLP techniques and machine learning models to preprocess the text data and build a predictive model for classifying tweets. The main steps of the project include:

1. Data Loading and Exploration: The notebook loads the training and test datasets and performs some initial data exploration.

2. Text Preprocessing: The text data undergoes several preprocessing steps to clean and prepare it for further analysis. These steps include removing HTML tags, URLs, emojis, punctuation, and handling special characters. The text is also converted to lowercase and lemmatized to simplify the analysis.

3. Feature Engineering: Extra features are added to the dataset based on the text data. These features include the number of characters, stopwords, punctuations, uppercase words, title case words, average word length, and counts of URLs, hashtags, and mentions.

4. Text Vectorization: The notebook uses the TF-IDF (Term Frequency-Inverse Document Frequency) vectorization technique to convert the processed text into numerical feature vectors.

5. Model Selection and Evaluation: Two machine learning models are employed for classification: Logistic Regression and RandomForestRegressor. The notebook uses K-Fold cross-validation to evaluate the performance of the models with different hyperparameters.

6. Model Training and Prediction: The best-performing model is trained on the entire training dataset and used to make predictions on the test dataset.

7. Submission: The final predictions are saved in a CSV file for submission.

## Libraries Used

The project utilizes various Python libraries for data processing, NLP, and machine learning, including:

- numpy
- pandas
- scikit-learn
- nltk
- spellchecker
- tqdm

## Dataset

The dataset used in this project is sourced from Kaggle and contains labeled tweets. The training dataset has columns for 'id', 'keyword', 'location', 'text', and 'target', where 'target' represents the class label (1 for real disaster tweet, 0 for non-disaster tweet). The test dataset has the same columns, except for the 'target', which is to be predicted.

## Model Selection

The notebook uses Logistic Regression and RandomForestRegressor models for classification. The hyperparameters are tuned using K-Fold cross-validation to find the best configuration.

## Results

The final model's performance is evaluated using the F1-score metric. The model with the highest F1-score is chosen for making predictions on the test dataset.

## Usage

To run the notebook, you will need to have Python installed along with the required libraries. You can execute the code cells sequentially to reproduce the preprocessing, feature engineering, model training, and prediction steps.

Please note that the notebook assumes the presence of the training and test datasets. If you want to reproduce the results, make sure to provide the correct file paths or modify the data loading code accordingly.

Feel free to explore the notebook and experiment with different models or feature engineering techniques to further improve the model's performance.

Happy coding!
