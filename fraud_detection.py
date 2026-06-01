# standard imports for the project
import streamlit as st 
import pandas as pd 
import joblib
import matplotlib.pyplot as plt 
import seaborn as sns

# load up the saved pipeline model
model = joblib.load("pickel_files/fraud_detection_pipeline.pkl")

# main app title and subheader setup
st.title("Fraud Detection App")
st.subheader("Sentinel Ai")
 
# quick text to show people how the app works
st.text("This app lets you test individual transactions or drop in a whole dataset to check for fraud using our backend model pipeline.")

# setting up tabs for layout options
tab1, tab2 = st.tabs(["Single Transaction", "Batch Upload"])
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


# Add the column 2 as well  
with tab2:
    #This is really a must inorder for he frauds to be identified
    st.text("Note: The CSV must contain these columns: type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, balanceDiffOrig, balanceDiffDest")
    st.subheader("Upload an transaction CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        st.text(f"Rows loaded {len(raw_df)}")

        feature_cols = [
            "type","amount","oldbalanceOrg","newbalanceOrig",
            "oldbalanceDest","newbalanceDest", "balanceDiffOrig", "balanceDiffDest"
        ]
        model_input = raw_df[feature_cols]

        batch_preds = model.predict(model_input)
        batch_probs = model.predict_proba(model_input)[:,1]

        results_df = raw_df.copy()


        results_df["Prediction"] = batch_preds
        results_df["Fraud_Probability"] = batch_probs.round(2)
        results_df["Status"] = results_df["Prediction"].map({0:"Legit",1:"Fraud"})

#Add the metrics in the COlumns
        col1,col2,col3 = st.columns(3) # FIX 1: changed st.column to st.columns
        col1.metric("Total Records", len(results_df))
        col2.metric("Fraud Detected", int(batch_preds.sum()))
        col3.metric("Legit Records", int(len(results_df)- batch_preds.sum()))

        st.markdown("----------")
        st.subheader("Feature Correlation Matrix")

        numeric_features = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "balanceDiffOrig", "balanceDiffDest"] # FIX 2: changed numeric_feature to numeric_features
        valid_numeric = [c for c in numeric_features if c in raw_df.columns]

        fig, ax = plt.subplots(figsize=(7, 4))
        sns.heatmap(raw_df[valid_numeric].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Heatmap")

        
        plt.tight_layout () 
        st.pyplot(fig)


        csv_data = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Results CSV",
            data=csv_data, 
            file_name="fraud_predictions.csv",
            mime="text/csv"
        )