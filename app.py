import nltk
import streamlit as st
import pickle
from nltk.corpus import stopwords
nltk.download('stopwords')
import string
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():  # for removing special ch only allow alphabets and numeric
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('spam_mail.pkl','rb'))

# Set page configuration
st.set_page_config(layout="wide", page_title="Email/SMS Spam Classifier", page_icon=":envelope:")


st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the Message")

if st.button('Predict'):
    #preprocessing
    transformed_sms = transform_text(input_sms)

    #vectorize
    vector_input = tfidf.transform([transformed_sms])

    #predict
    result = model.predict(vector_input)[0]

    #show
    if result==1:
        st.header("Spam")
        st.markdown("<span style='color:red'>This message is classified as spam.</span>", unsafe_allow_html=True)

    else:
        st.header("Not Spam")
        st.markdown("<span style='color:green'>This message is not classified as spam.</span>", unsafe_allow_html=True)
