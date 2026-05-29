#Import the neccesary libraries
import streamlit as st 
import pandas as pd 
import joblib
import matplotlib.pyplot as plt 
import seaborn as sns

#Load the pickel file
model = joblib.load("pickel_files/fraud_detection_pipeline.pkl")

#Design the Frontend part
st.title("Fraud Detection App")
st.subheader("Sentinel Ai")

#. Add two columns for single transaction and 
tab1, tab2 = st.tabs(["Single Transaction","Batch Upload"])

# Design the single tab one
with tab1:
    st.subheader("Test onee transaction")
    
    # split layout into 2 parts so it doesnt look super long
    column1, column2 = st.columns(2)

    with column1:
        amount = st.number_input("Amount", min_value=0.0, value=500.0)
        # fixed the 'min_values' typo from earlier to make it work!
        oldbalanceOrg = st.number_input("Old Balance (Origin)", min_value=0.0, value=1000.0)
        newbalanceOrig = st.number_input("New Balance(Origin)", min_value=0.0, value=500.0)
        oldbalanceDest = st.number_input("Old Balance (Destination)", min_value=0.0, value=0.0)

    with column2:
        newbalanceDest = st.number_input("New Balance (Destination)", min_value=0.0, value=500.0)
        balanceDiffOrig = st.number_input("Balance Diff (Origin)", value=500.0)
        balanceDiffDest = st.number_input("Balance Diff (Destination)", value=500.0)
        txn_type = st.selectbox("Transaction Type", ["CASH_OUT", "TRANSFER", "PAYMENT", "CASH_IN", "DEBIT"])

    # Triggering  the model prediction on click
    if st.button("Predict"):
        # raw dictionary converted to dataframe right here
        my_test_data = pd.DataFrame([{
            "type": txn_type,
            "amount": amount,
            "oldbalanceOrg": oldbalanceOrg,
            "newbalanceOrig": newbalanceOrig,
            "oldbalanceDest": oldbalanceDest,
            "newbalanceDest": newbalanceDest,
            "balanceDiffOrig": balanceDiffOrig,
            "balanceDiffDest": balanceDiffDest,
        }])

        # get prediction arrays out
        final_pred = model.predict(my_test_data)[0]
        final_prob = model.predict_proba(my_test_data)[0][1]

        # showing result alerts based on prediction binary by the model
        if final_pred == 1:
            st.error("Fraud Detected")
        else:
            st.success("Transaction looks legit")

        st.metric("Fraud Probability", f"{final_prob:.2%}")


# Add the column 2 as well now   
