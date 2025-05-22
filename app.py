import streamlit as st
import pandas as pd
import joblib
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Downloads for nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the trained model
model = joblib.load("email_spam_classifier.pkl")

# Preprocessing function
stop_words = set(stopwords.words('english'))
punctuations = set(string.punctuation)
wl = WordNetLemmatizer()

def cleaning(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words and word not in punctuations]
    tokens_lem = [wl.lemmatize(word) for word in tokens]
    return ' '.join(tokens_lem)

# Streamlit App UI
st.title("Email Spam Classifier")
st.write("This app predicts whether an email message is **Spam** or **Not Spam**.")

user_input = st.text_area("Enter your email message here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        prediction = model.predict([user_input])
        result = "Not Spam (Ham)" if prediction[0] == 1 else "Spam"
        st.success(f"Prediction: {result}")
