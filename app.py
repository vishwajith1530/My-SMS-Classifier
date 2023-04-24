import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def transform_mssg(mssg):
    mssg = mssg.lower()
    mssg = nltk.word_tokenize(mssg)

    z = []
    for i in mssg:
        if i.isalnum():
            z.append(i)

    mssg = z[:]
    z.clear()

    for i in mssg:
        if i not in stopwords.words('english') and i not in string.punctuation:
            z.append(i)

    mssg = z[:]
    z.clear()

    for i in mssg:
        z.append(ps.stem(i))

    return " ".join(z)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_mssg(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")