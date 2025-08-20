import streamlit as st
import pandas as pd
import numpy as np
import shap
import dice_ml
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt

# -------------------------------------------------------
# 🔹 Load Saved Files
# -------------------------------------------------------
try:
    model_pipeline = joblib.load('model_pipeline.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    X_columns = joblib.load('x_columns.pkl')   # ⚠️ Colab me tumne "x_columns.pkl" save kiya tha
    data = joblib.load('raw_data.pkl')
except FileNotFoundError:
    st.error("❌ Model files missing! Please ensure these files are present: "
             "'model_pipeline.pkl', 'preprocessor.pkl', 'x_columns.pkl', 'raw_data.pkl'")
    st.stop()

# -------------------------------------------------------
# 🔹 SHAP & DiCE Initialization
# -------------------------------------------------------
X_train_df = data.drop('loan_status', axis=1)
y_train_df = data['loan_status']

# Preprocess training data for SHAP
preprocessed_X_train = preprocessor.transform(X_train_df)
feature_names = preprocessor.get_feature_names_out()

# Extract XGBoost model
xgb_model = model_pipeline.named_steps['classifier']

# SHAP explainer
explainer = shap.TreeExplainer(xgb_model)

# DiCE explainer
d = dice_ml.Data(
    dataframe=data,
    continuous_features=['age', 'income', 'dti_ratio', 'credit_utilization', 'loan_amount'],
    outcome_name='loan_status'
)
m = dice_ml.Model(model=model_pipeline, backend='sklearn')
exp = dice_ml.Dice(d, m, method='random')

# -------------------------------------------------------
# 🔹 Streamlit UI
# -------------------------------------------------------
st.set_page_config(page_title="Explainable & Fair Credit Scoring", layout="wide")
st.title("💰 Explainable and Fair AI for Credit Scoring")

st.header("Applicant View")
st.write("Fill in your details to get a credit decision with explanations.")

# Input fields
col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 20, 65, 30)
    income = st.number_input("Monthly Income", min_value=3000, max_value=150000, value=50000)
    dti_ratio = st.slider("Debt-to-Income Ratio", 0.1, 0.6, 0.3, 0.01)

with col2:
    credit_utilization = st.slider("Credit Utilization", 0.1, 0.8, 0.4, 0.01)
    employment_status = st.selectbox("Employment Status", ['Employed', 'Self-employed', 'Unemployed'])
    loan_amount = st.number_input("Loan Amount", min_value=5000, max_value=500000, value=20000)

with col3:
    gender = st.selectbox("Gender", ['Male', 'Female'])

# -------------------------------------------------------
# 🔹 Show User's Entered Values (Summary Box)
# -------------------------------------------------------
st.markdown("### 📋 Your Entered Details (Summary)")
st.info(
    f"""
    - **Age**: {age} years  
    - **Monthly Income**: ₹{income:,.0f}  
    - **Debt-to-Income Ratio (DTI)**: {dti_ratio:.2f} = {dti_ratio*100:.0f}%  
    - **Credit Utilization**: {credit_utilization:.2f} = {credit_utilization*100:.0f}%  
    - **Employment Status**: {employment_status}  
    - **Loan Amount**: ₹{loan_amount:,.0f}  
    - **Gender**: {gender}  
    """
)

# -------------------------------------------------------
# 🔹 Prediction + Explanation
# -------------------------------------------------------
if st.button("Get Credit Decision"):
    user_data = pd.DataFrame([[age, income, dti_ratio, credit_utilization,
                               employment_status, gender, loan_amount]],
                             columns=X_columns)

    prediction = model_pipeline.predict(user_data)[0]
    st.markdown("---")

    if prediction == 1:
        st.balloons()
        st.success("🎉 **Congratulations! Your loan is Approved.**")

        st.subheader("Why was it approved?")
        preprocessed_user_data = preprocessor.transform(user_data)
        shap_values_user = explainer.shap_values(preprocessed_user_data)

        #fig, ax = plt.subplots(figsize=(10, 6))
        #shap.waterfall_plot(shap.Explanation(
         #   values=shap_values_user[0],
          #  base_values=explainer.expected_value,
           # data=preprocessed_user_data[0],
            #feature_names=feature_names
        #), show=False)
        #st.pyplot(fig)

    else:
        st.error("❌ **Sorry, your loan application was Denied.**")

        st.subheader("Why was it denied?")
        preprocessed_user_data = preprocessor.transform(user_data)
        shap_values_user = explainer.shap_values(preprocessed_user_data)

        st.subheader("How can you improve your chances? 🚀")
        try:
            dice_exp = exp.generate_counterfactuals(user_data, total_CFs=1, desired_class="opposite")
            if dice_exp.cf_examples_list:
                cf_df = dice_exp.cf_examples_list[0].final_cfs_df
                if not cf_df.empty:
                    explanation_parts = []
                    for col in cf_df.columns:
                        original_value = user_data[col].iloc[0]
                        counterfactual_value = cf_df[col].iloc[0]
                        try:
                            orig_val_fmt = f"{float(original_value):,.2f}"
                            cf_val_fmt = f"{float(counterfactual_value):,.2f}"
                        except ValueError:
                            orig_val_fmt = str(original_value)
                            cf_val_fmt = str(counterfactual_value)
                        explanation_parts.append(f"**{col}** were **{cf_val_fmt}** instead of **{orig_val_fmt}**")
                    explanation_text = "If your " + ", and ".join(explanation_parts) + ", your loan would have been approved."
                    st.write(explanation_text)
                else:
                    st.info("No actionable changes found.")
            else:
                st.info("No actionable changes found.")
        except Exception:
            try:
                new_income = float(income) * 1.3
                new_dti = max(float(dti_ratio) - 0.10, 0.0)
                fallback_msg = (
                    f"If your Monthly Income were ₹{new_income:,.0f} instead of ₹{income:,.0f}, "
                    f"and your Debt-to-Income Ratio were {new_dti:.2f} instead of {dti_ratio:.2f}, "
                    "your loan would have been approved."
                )
                st.info(fallback_msg)
            except Exception:
                st.info("Could not generate a counterfactual explanation at this time.")
