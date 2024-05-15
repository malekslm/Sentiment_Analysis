import streamlit as st
import joblib
import speech_recognition as sr
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model and vectorizer
model = joblib.load('svm_model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')

# Function to transcribe audio to text
def audio_to_text():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Say something...")
        audio = r.listen(source)
    try:
        text = r.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Sorry, could not understand audio."
    except sr.RequestError as e:
        return "Could not request results; {0}".format(e)

# Function to predict sentiment and return probabilities
def predict_sentiment(tweet):
    # Handle empty input
    if not tweet.strip():
        return 'Please enter a tweet.', None

    # Vectorize the input tweet
    vectorized_tweet = vectorizer.transform([tweet])
    # Predict sentiment probabilities
    probabilities = model.predict_proba(vectorized_tweet).flatten()
    sentiment = model.predict(vectorized_tweet)[0]
    return sentiment, probabilities

# Streamlit UI
st.title('Sentiment Analysis')
option = st.radio("Select Input Source:", ("Text", "Audio"))
if option == 'Text':
    tweet_input = st.text_input('Enter your tweet:')
elif option == 'Audio':
    tweet_input = audio_to_text()

if st.button('Predict'):
    sentiment, probabilities = predict_sentiment(tweet_input)
    if sentiment == 'Please enter a tweet.':
        st.write(sentiment)
    else:
        if sentiment == 1:
            st.write('Positive Sentiment')
        else:
            st.write('Negative Sentiment')

        # Display probability scores
        st.write('Probability of Positive Sentiment:', probabilities[1])
        st.write('Probability of Negative Sentiment:', probabilities[0])

# Display the transcribed text if input source is audio
if option == 'Audio':
    st.subheader("Transcribed Text:")
    st.write(tweet_input)
