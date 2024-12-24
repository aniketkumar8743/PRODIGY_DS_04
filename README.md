# Twitter Sentiment Analysis
This repository contains a Twitter Sentiment Analysis project that classifies Twitter messages into positive, negative, or neutral categories. The model uses a Random Forest Classifier with a TF-IDF Vectorizer pipeline to predict the sentiment of tweets. The project achieves an accuracy of 91.67% on the sentiment classification task.

Additionally, a Graphical User Interface (GUI) has been developed that allows users to input any tweet and receive real-time sentiment predictions.

## Key Features
**Sentiment Classification**: Classifies tweets into three categories: Positive, Negative, and Neutral.
**Random Forest Classifier**: Uses a Random Forest model for classification, known for its robustness and accuracy.
**TF-IDF Vectorizer**: Converts textual data into numerical format using Term Frequency-Inverse Document Frequency (TF-IDF) to help the model understand the importance of words.
**GUI**: A simple, user-friendly interface to enter Twitter messages and classify their sentiments interactively.
**High Accuracy**: The model achieves 91.67% accuracy on test data, making it a reliable sentiment classifier for Twitter data.

## Technologies Used
**Python**: The primary programming language used for implementing the sentiment analysis system.
**Scikit-learn**: For building the Random Forest Classifier and performing other machine learning tasks.
**TF-IDF Vectorizer**: A method for converting text data into numerical features for machine learning models.
**Tkinter**: Used to build the Graphical User Interface (GUI) for real-time sentiment predictions.
**Pandas**: For data handling and manipulation.
**Matplotlib/Seaborn**: For visualizing model performance (optional).

## Dataset
The model uses a dataset of labeled tweets that contain positive, negative, or neutral sentiments. Each tweet is labeled accordingly, which allows the model to learn to classify unseen tweets. The dataset includes:

**Tweet Text**: The content of the tweet.
**Sentiment Label**: The sentiment of the tweet, categorized as positive, negative, or neutral.
## Model Architecture
### 1. Data Preprocessing:

- The dataset is preprocessed by cleaning the text, removing special characters, stop words, and converting text to lowercase to standardize it.
- The TF-IDF Vectorizer is used to convert the text data into numerical format, which is then fed into the machine learning model.
### 2. Model Training:

- The Random Forest Classifier is used for training the model. It is an ensemble method that combines multiple decision trees to improve performance and reduce overfitting.
- The model is trained on the TF-IDF features extracted from the text data, and it predicts the sentiment of each tweet.
### 3. Model Evaluation:

- The model is evaluated using standard metrics like accuracy. It achieves an accuracy of 91.67%, which indicates that the model is performing well in classifying the sentiment of tweets.
- The model is tested on a separate validation set to ensure it generalizes well to unseen data.

## Code Explanation
### 1. Preprocessing the Data
The data is cleaned and prepared for model training. This includes removing irrelevant characters (like URLs, mentions, and hashtags), tokenizing the text, and applying TF-IDF Vectorization to convert the text into a numerical format suitable for machine learning.

### 2. TF-IDF Vectorization
python
Copy code
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(tweets_data['tweet'])
The TF-IDF vectorizer transforms each tweet into a vector that captures the importance of each word in the context of the entire dataset. The model uses these vectors to classify the sentiment of tweets.

### 3. Training the Random Forest Classifier
python
Copy code
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
The Random Forest Classifier is trained on the transformed text data, which consists of TF-IDF features. This model is well-suited for handling high-dimensional data and provides robust predictions.

### 4. Prediction with the Model
python
Copy code
predictions = clf.predict(X_test)
Once trained, the model predicts the sentiment of unseen tweets, classifying them as either positive, negative, or neutral.

## Results
The Random Forest Classifier achieves an accuracy of 91.67% on the test data, indicating that the model is able to accurately predict the sentiment of Twitter messages.

## Contributing
Contributions are welcome! Feel free to fork the repository, make changes, and submit a pull request.
