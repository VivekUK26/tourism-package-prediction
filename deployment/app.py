import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download


HF_USERNAME = "AiRemastered"  # MY HUGGING FACE USERNAME

MODEL_REPO = f"{HF_USERNAME}/tourism-package-model"

# Load model and encoders from Hugging Face Hub
@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id=MODEL_REPO, filename="best_model.pkl", repo_type="model")
    encoders_path = hf_hub_download(repo_id=MODEL_REPO, filename="label_encoders.pkl", repo_type="model")
    model = joblib.load(model_path)
    encoders = joblib.load(encoders_path)
    return model, encoders

model, label_encoders = load_model()

# App title and description
st.title("Tourism Package Prediction")
st.markdown("### Predict if a customer will purchase the Wellness Tourism Package")
st.markdown("---")

# Create input form
st.sidebar.header("Customer Information")

# Input fields
age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=35)
type_of_contact = st.sidebar.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
city_tier = st.sidebar.selectbox("City Tier", [1, 2, 3])
duration_of_pitch = st.sidebar.number_input("Duration of Pitch (minutes)", min_value=1, max_value=60, value=15)
occupation = st.sidebar.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
num_persons_visiting = st.sidebar.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
num_followups = st.sidebar.number_input("Number of Follow-ups", min_value=1, max_value=10, value=3)
product_pitched = st.sidebar.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
preferred_property_star = st.sidebar.selectbox("Preferred Property Star", [3, 4, 5])
marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
num_trips = st.sidebar.number_input("Number of Trips (annually)", min_value=1, max_value=20, value=2)
passport = st.sidebar.selectbox("Has Passport?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
pitch_satisfaction_score = st.sidebar.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
own_car = st.sidebar.selectbox("Owns Car?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
num_children_visiting = st.sidebar.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0)
designation = st.sidebar.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
monthly_income = st.sidebar.number_input("Monthly Income", min_value=10000, max_value=100000, value=25000)

# Create prediction button
if st.sidebar.button("Predict", type="primary"):
    # Create input dataframe
    input_data = pd.DataFrame({
        'Age': [age],
        'TypeofContact': [type_of_contact],
        'CityTier': [city_tier],
        'DurationOfPitch': [duration_of_pitch],
        'Occupation': [occupation],
        'Gender': [gender],
        'NumberOfPersonVisiting': [num_persons_visiting],
        'NumberOfFollowups': [num_followups],
        'ProductPitched': [product_pitched],
        'PreferredPropertyStar': [preferred_property_star],
        'MaritalStatus': [marital_status],
        'NumberOfTrips': [num_trips],
        'Passport': [passport],
        'PitchSatisfactionScore': [pitch_satisfaction_score],
        'OwnCar': [own_car],
        'NumberOfChildrenVisiting': [num_children_visiting],
        'Designation': [designation],
        'MonthlyIncome': [monthly_income]
    })

    # Encode categorical variables
    categorical_cols = ['TypeofContact', 'Occupation', 'Gender', 'ProductPitched', 'MaritalStatus', 'Designation']
    for col in categorical_cols:
        if col in label_encoders:
            input_data[col] = label_encoders[col].transform(input_data[col])

    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    # Display result
    st.markdown("## Prediction Result")

    if prediction == 1:
        st.success("The customer is likely to PURCHASE the Wellness Tourism Package!")
        st.balloons()
    else:
        st.warning("The customer is NOT likely to purchase the package.")

    # Display probabilities
    st.markdown("### Prediction Probabilities")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Not Purchase", f"{probability[0]*100:.1f}%")
    with col2:
        st.metric("Purchase", f"{probability[1]*100:.1f}%")

    # Show input summary
    st.markdown("---")
    st.markdown("### Input Summary")
    st.dataframe(input_data.T.rename(columns={0: "Value"}))

# Footer
st.markdown("---")
st.markdown("*Built with Streamlit | Tourism Package Prediction Model*")
