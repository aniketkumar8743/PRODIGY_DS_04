import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')

ps = PorterStemmer()
model = pickle.load(open('RFC_Model', 'rb'))

st.title("Attitudes Towards Specific Topics Or Brands")
input_sms = st.text_area("Enter the Message")

def transform_text(text):
    text = text.lower()
    text = nltk.sent_tokenize(text)
    stem = []
    for sent in text:
        sent = sent.translate(str.maketrans('', '', string.punctuation))
        words = nltk.word_tokenize(sent)
        words = [word for word in words if word not in stopwords.words('english')]
        words = [ps.stem(word) for word in words]
        stem.append(' '.join(words))
    return ' '.join(stem)

if st.button('Predict'):
    transform_sms = transform_text(input_sms)
    vector_input = [transform_sms]
    result = model.predict(vector_input)[0]
    if result == 0:
        st.header("Negative")
    elif result == 1:
        st.header("Positive")
    elif result == 2:
        st.header("Neutral")
    elif result == 3:
        st.header("Irrelevant")