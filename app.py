#this app is live on "https://creditscoringtoolbyabhaykumar.streamlit.app/"

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mymodel import predict  

# Feature names
feature_names = [
    "Checking Account Status", "Loan Duration (months)", "Credit History", "Loan Purpose", "Credit Amount",
    "Savings Account/Bonds", "Employment Duration", "Installment Rate (% of Income)", "Personal Status and Sex",
    "Other Debtors/Guarantors", "Residence Duration", "Property Type", "Age (years)", "Other Installment Plans",
    "Housing", "Existing Credits", "Job", "Number of Dependents", "Telephone", "Foreign Worker",
    "Unknown Feature 1", "Unknown Feature 2", "Unknown Feature 3", "Unknown Feature 4"
]

#  categorical features
dropdown_features = {
    "Checking Account Status": {1: "A11", 2: "A12", 3: "A13", 4: "A14"},
    "Credit History": {0: "No credit", 1: "Paid in full", 2: "All paid", 3: "Delay", 4: "Critical"},
    "Loan Purpose": {0: "New car", 1: "Used car", 2: "Furniture", 3: "Radio/TV", 4: "Domestic", 5: "Repairs",
                     6: "Education", 7: "Vacation", 8: "Retraining", 9: "Business"},
    "Savings Account/Bonds": {1: "<100", 2: "100-500", 3: "500-1000", 4: ">=1000", 5: "Unknown"},
    "Employment Duration": {1: "<1yr", 2: "1-4yrs", 3: "4-7yrs", 4: ">=7yrs", 5: "Unemployed"},
    "Personal Status and Sex": {1: "Male-divorced", 2: "Female-married", 3: "Male-single", 4: "Male-married", 5: "Female-single"},
    "Other Debtors/Guarantors": {1: "None", 2: "Co-applicant", 3: "Guarantor"},
    "Property Type": {1: "Real estate", 2: "Life insurance", 3: "Car", 4: "Unknown"},
    "Other Installment Plans": {1: "Bank", 2: "Stores", 3: "None"},
    "Housing": {1: "Own", 2: "Rent", 3: "Free"},
    "Job": {1: "Unemployed", 2: "Unskilled", 3: "Skilled", 4: "Management"},
    "Telephone": {1: "None", 2: "Yes"},
    "Foreign Worker": {1: "Yes", 2: "No"},
}

# UI Layout
st.title("💳 Credit Risk Predictor")
st.markdown("Predict if a person is a **Good** or **Bad** credit risk based on financial details.")

st.subheader("Enter Customer Details")

user_input = []

# Collect user input
for i, feature in enumerate(feature_names):
    if feature in dropdown_features:
        label_map = dropdown_features[feature]
        display = st.selectbox(feature, list(label_map.values()), key=feature)
        selected_key = [k for k, v in label_map.items() if v == display][0]
        user_input.append(selected_key)
    else:
        default = 12 if "months" in feature else 1000 if "Amount" in feature else 1
        val = st.number_input(feature, value=float(default), key=feature)
        user_input.append(val)

# Prediction and visualization
if st.button("Predict Credit Risk"):
    input_df = pd.DataFrame([user_input], columns=[f"Feature_{i}" for i in range(1, 25)])
    prediction, model = predict(input_df)

    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.success("✅ Prediction: Good Credit Risk")
    else:
        st.error("❌ Prediction: Bad Credit Risk")

    # Feature importance plot
    st.subheader("📊 Top 10 Important Features")

    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by="Importance", ascending=False).head(10)

    
    sns.set_style("whitegrid")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x="Importance",
        y="Feature",
        data=feature_importance_df,
        palette="crest"
    )

    # Styling
    ax.set_title("🔍 Top 10 Most Important Features", fontsize=16, weight='bold', pad=15)
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    ax.tick_params(axis='both', labelsize=10)
    sns.despine(left=True, bottom=True)

    st.pyplot(fig)

st.markdown("> ⚠️ *Note: This app is built for learning purposes using the German Credit dataset (numeric format).*")
