
import pickle
import streamlit as stm
import numpy as np

# Load the saved model and scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
    
#Streamlit App Interface(UI)
stm.title("Titanic Survival Predictor")
stm.write("Enter Passanger input for prediction")

# User Inputs
pclass = stm.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
age = stm.slider("Age", 0, 80, 30)
sibsp = stm.slider("Number of Siblings/Spouses Aboarded", 0, 8, 0)
parch = stm.slider("Number of Parents/Children Aboarded", 0, 6, 0)
fare = stm.number_input("Fare Paid", min_value=0.0, value=32.0)
sex = stm.radio("Sex", ['Male', 'Female'])
embarked = stm.radio("Port of Embarkation", ['C', 'Q', 'S'])

# Convert categorical inputs to dummies (like in training)
sex_male = 1 if sex == 'Male' else 0
embarked_q = 1 if embarked == 'Q' else 0
embarked_s = 1 if embarked == 'S' else 0

# Create input array for model (order must match training!)
input_data = np.array([[pclass, age, sibsp, parch, fare, sex_male, embarked_q, embarked_s]])
input_scaled = scaler.transform(input_data)

# Predict
if stm.button("Predict Survival"):
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]
    
    if prediction == 1:
        stm.success(f"The passenger is likely to **SURVIVE**  (Probability: {probability:.2%})")
    else:
        stm.error(f"The passenger is likely to **NOT survive** (Probability: {probability:.2%})")
