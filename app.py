# app.py

import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from PIL import Image  # For displaying images

# Title and description for the app
st.title("NLP Project for Disaster Tweet Classification")

# Display the dashboard banner (image_disaster.png)
banner_image = Image.open('Notebook/images/image_disaster.png')
st.image(banner_image, caption="Disaster Tweet Classification Dashboard", use_container_width=True)

# Description of the app
st.markdown("""
    This is the **NLP Project for Disaster Tweet Classification** using a **Logistic Regression** model.
    The app classifies tweets into **Disaster** or **Not Disaster** and displays relevant images based on the classification result.
""")

# Load the Logistic Regression model
with open('Notebook/best_logistic_model.pkl', 'rb') as file:
    lr_model = pickle.load(file)

# Load the TfidfVectorizer used in training
with open('Notebook/tfidf_vectorizer.pkl', 'rb') as file:
    tfidf_vectorizer = pickle.load(file)

# Preprocessing function for Logistic Regression (TF-IDF vectorization)
def preprocess_for_lr(text):
    # Convert text to TF-IDF features
    text_tfidf = tfidf_vectorizer.transform([text])
    return text_tfidf

# Predict function for Logistic Regression
def predict_lr(text):
    # Preprocess text and predict
    text_tfidf = preprocess_for_lr(text)
    prediction = lr_model.predict(text_tfidf)
    return "Disaster" if prediction[0] == 1 else "Not Disaster"

# Input text area for tweet input
input_text = st.text_area("Enter the tweet here:")

# Button to trigger prediction
if st.button("Predict"):
    if input_text:
        # Get prediction from Logistic Regression
        prediction = predict_lr(input_text)
        
        # Show prediction result
        st.write(f"### Prediction: {prediction}")

        # Display images based on prediction
        if prediction == "Disaster":
            disaster_image = Image.open('Notebook/images/images_dis.jpg')
            st.image(disaster_image, caption="Disaster Tweet", use_container_width=True)
        else:
            non_disaster_image = Image.open('Notebook/images/img_non.png')
            st.image(non_disaster_image, caption="Non-Disaster Tweet", use_container_width=True)

        # Show explanation for classification
        if prediction == "Disaster":
            st.success("This tweet is classified as **Disaster**.")
        else:
            st.warning("This tweet is classified as **Not Disaster**.")
    else:
        st.error("Please enter a tweet for classification.")

# Ending portion of the app
st.markdown("""
    ### Project by Minal Devikar
    This project aims to classify tweets based on their content into **Disaster** or **Not Disaster** using **Natural Language Processing (NLP)** techniques.
""")
