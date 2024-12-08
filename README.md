# NLP Project for Disaster Tweet Classification

This project classifies tweets into **Disaster** or **Not Disaster** using a **Logistic Regression** model. The app allows users to input a tweet, and it classifies it based on the content. The app displays relevant images depending on whether the tweet is classified as **Disaster** or **Not Disaster**.

## Project Description

The goal of this project is to classify tweets as either **Disaster** or **Not Disaster** based on their content using **Natural Language Processing (NLP)** techniques. The app uses the following models:
- **Logistic Regression (ML model)**: For classifying tweets.
- **Artificial Neural Network (ANN)**: This model is also available but we focus on the **Logistic Regression** model in this example.

## Features

- **Banner Image**: Displays a banner image on the homepage.
- **Text Input**: Users can enter a tweet for classification.
- **Image Display**: Based on the classification, the app will display the appropriate image.
  - If classified as **Disaster**, an image related to disaster is shown.
  - If classified as **Not Disaster**, a different image is displayed.

## Pipeline

The **pipeline** for this project involves several key stages from data processing to model deployment. Below is the step-by-step breakdown of the pipeline:

### 1. **Data Collection**
   - The dataset used in this project contains tweets labeled as either **Disaster** or **Not Disaster**.
   - The dataset is preprocessed for cleaning and feature extraction.

### 2. **Data Preprocessing**
   - **Text Cleaning**: Raw tweet text is cleaned by removing stop words, special characters, and URLs.
   - **Tokenization**: The text data is split into tokens (words).
   - **TF-IDF Vectorization**: 
     - The **TF-IDF Vectorizer** is used to convert the cleaned and tokenized text data into numeric features. This step transforms the tweet text into a feature matrix with a fixed number of features (max 6400).
     - The **TfidfVectorizer** is trained on the entire dataset and then used to transform new tweet text into the same feature space during prediction.

### 3. **Model Training**
   - **Logistic Regression Model**:
     - The **Logistic Regression** algorithm is used to train the model on the vectorized tweet data.
     - The model learns to classify tweets based on the features extracted by the **TF-IDF Vectorizer**.
   - **Model Evaluation**:
     - The model is evaluated using **accuracy**, **precision**, **recall**, and **F1-score** to determine its performance.

### 4. **Model Deployment**
   - The **trained Logistic Regression model** is saved using **pickle** to ensure it can be used for predictions in the Streamlit app.
   - The **TF-IDF Vectorizer** is also saved to ensure consistency during preprocessing in the deployed app.
   
### 5. **Streamlit App**
   - The **Streamlit app** is built to allow users to input a tweet and see whether it is classified as **Disaster** or **Not Disaster**.
   - The app:
     - Accepts input tweets from the user.
     - Applies the same **TF-IDF transformation** to the input text as was done during training.
     - Uses the **Logistic Regression model** to classify the tweet.
     - Displays relevant images based on the classification result (Disaster or Non-Disaster).

### 6. **Image Display**
   - **Images for Disaster and Non-Disaster Predictions**:
     - If the tweet is classified as **Disaster**, an image related to disaster is displayed.
     - If the tweet is classified as **Not Disaster**, an image showing a peaceful or non-disaster context is displayed.
     - ### Output Display
![Output Image 1](https://github.com/minalmmm/NLP-Project-for-Disaster-Tweet-Classification/blob/main/Images/img1.png)  
![Output Image 2](https://github.com/minalmmm/NLP-Project-for-Disaster-Tweet-Classification/blob/main/Images/img2.png)  
![Output Image 3](https://github.com/minalmmm/NLP-Project-for-Disaster-Tweet-Classification/blob/main/Images/img3.png)

