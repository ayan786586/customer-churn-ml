
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_logistic_pipeline.joblib"


st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Customer Churn Prediction")
st.write(
    "Enter customer details to estimate the likelihood "
    "of the customer leaving the company."
)


@st.cache_resource
def load_model():
    """Load the saved preprocessing and prediction pipeline."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Make sure the trained pipeline exists in the models folder."
        )

    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as error:
    st.error(f"Unable to load the model: {error}")
    st.stop()


with st.form("customer_form"):
    st.subheader("Customer Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox(
            "Senior Citizen", [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
        )
        partner = st.selectbox("Has Partner?", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
        tenure = st.number_input(
            "Tenure (months)", min_value=0, max_value=72, value=12
        )
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox(
            "Multiple Lines", ["No", "Yes", "No phone service"]
        )

    with col2:
        internet_service = st.selectbox(
            "Internet Service", ["DSL", "Fiber optic", "No"]
        )
        online_security = st.selectbox(
            "Online Security", ["Yes", "No", "No internet service"]
        )
        online_backup = st.selectbox(
            "Online Backup", ["Yes", "No", "No internet service"]
        )
        device_protection = st.selectbox(
            "Device Protection", ["Yes", "No", "No internet service"]
        )
        tech_support = st.selectbox(
            "Tech Support", ["Yes", "No", "No internet service"]
        )
        streaming_tv = st.selectbox(
            "Streaming TV", ["Yes", "No", "No internet service"]
        )
        streaming_movies = st.selectbox(
            "Streaming Movies", ["Yes", "No", "No internet service"]
        )

    with col3:
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"],
        )
        paperless_billing = st.selectbox(
            "Paperless Billing", ["Yes", "No"]
        )
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=70.0,
            step=1.0,
        )
        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=840.0,
            step=10.0,
        )

    submitted = st.form_submit_button(
        "Predict Churn",
        use_container_width=True,
    )


if submitted:
    customer_data = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    customer_df = pd.DataFrame([customer_data])

    try:
        prediction = int(model.predict(customer_df)[0])
        probability = float(model.predict_proba(customer_df)[0, 1])

        st.divider()
        st.subheader("Prediction Result")

        result_col, probability_col = st.columns(2)

        with result_col:
            if prediction == 1:
                st.warning("Prediction: Customer likely to churn")
            else:
                st.success("Prediction: Customer likely to stay")

        with probability_col:
            st.metric(
                "Estimated Churn Probability",
                f"{probability:.1%}",
            )

        st.caption(
            "This is a machine-learning estimate, not a guarantee "
            "of future customer behavior."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")
