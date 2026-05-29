# Sentinel Ai(Banking Fraud Detector)

This is an machine learning project I built to detect the fradulent in all of the financial transactions using a trained classification model.

I used Streamlit to delpoy this and this is actually my second project i built using streamlit

This project has taught me a lot the professional way of commiting the code and structuring the code

## Dataset

I trained this model using the data with 6.3 million rows found in kaggle you can download it from the link below:
 https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset?resource=download
 
To use it locally, anyone can download rename it as raw.csv and add it inside the data/raw folder.

## How to run:
1. Install all teh required libraries: streamlit pandas scikit-learn joblib matplotlib seaborn

2. Run the app using the code streamlit run yourfilename.py

## Features

- Single transaction testing through a simple input form
- Batch processing by uploading a CSV file in the site
- Summary counter showing total, fraud, and actual legit records
- Correlation heatmap of the input features as well

## AI Use Declaration

Since i was new to teh streamlit i had to ask AI about some functions i could use some of them being like .markdown and .subheader. Also I asked the AI to make two trial csv so that we can test our model's batch processing which are inside this data/trail

Other than that Gemini has acted as an coach teaching me ways to improve my committing structure and frequency and help me debug some of my code