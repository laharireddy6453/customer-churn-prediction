import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load("churn_model_lahari.pkl")

# Set page config
st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

# App title
st.title("📉 Customer Churn Prediction App")
st.write("Upload customer data or fill the form below to predict churn.")



# Sidebar navigation
option = st.sidebar.radio("Choose Input Type:", ["📁 Upload CSV", "📝 Manual Entry"])
if option == "📂 Upload CSV":
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        st.write("📄 Preview of Uploaded Data:")
        st.dataframe(input_df)

        if st.button("Predict"):
            prediction, probability = predict(input_df)
            st.write("🔮 Prediction:", prediction)
            st.write("📊 Churn Probability:", probability)

elif option == "📝 Manual Entry":
    st.number_input(...)  # Or st.text_input etc.

# Expected input features (update based on your dataset)
input_features = ['Age', 'Gender_Female', 'Location_New York', 'Subscription_Length_Months',
                  'Monthly_Bill', 'Total_Usage_GB']


def predict(df):
    try:
        prediction = model.predict(df)
        probability = model.predict_proba(df)[:, 1]
        return prediction, probability
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        return None, None





# --- 1. CSV Upload Mode ---
if option == "📁 Upload CSV":
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file:
        input_df = pd.read_csv(uploaded_file)
        st.write("🔍 Preview of Uploaded Data:")
        st.dataframe(input_df)

        if st.button("Predict Churn"):
            prediction, probability = predict(input_df)
            results = input_df.copy()
            results["Prediction"] = ["Churn" if p == 1 else "Not Churn" for p in prediction]
            results["Churn Probability"] = np.round(probability, 2)
            st.write("📊 Prediction Results:")
            st.dataframe(results)

# --- 2. Manual Input Mode ---
else:
    st.subheader("Enter Customer Info:")

    age = st.slider("Age", 18, 70, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    location = st.selectbox("Location", ["New York", "Los Angeles", "Miami", "Houston", "Chicago"])
    sub_len = st.slider("Subscription Length (months)", 1, 60, 12)
    monthly_bill = st.number_input("Monthly Bill ($)", 10, 500, 60)
    total_usage = st.number_input("Total Usage (GB)", 1, 1000, 100)

    # One-hot encoding manually
    input_dict = {
        "Age": age,
        "Gender_Female": 1 if gender == "Female" else 0,
        "Location_New York": 1 if location == "New York" else 0,
        "Subscription_Length_Months": sub_len,
        "Monthly_Bill": monthly_bill,
        "Total_Usage_GB": total_usage
    }

    input_df = pd.DataFrame([input_dict])

    if st.button("Predict Churn"):
        prediction, probability = predict(input_df)
        st.success(f"Prediction: **{'Churn' if prediction[0]==1 else 'Not Churn'}**")
        st.info(f"Churn Probability: **{round(probability[0]*100, 2)}%**")
