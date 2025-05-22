import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('punkt')       
nltk.download('stopwords')    
nltk.download('wordnet')     


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
