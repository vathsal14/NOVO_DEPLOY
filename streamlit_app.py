import streamlit as st
import numpy as np
import pickle

# Load the trained model and scalers
model_path = "model/Parkinson_Model.pkl"
scaler_X_path = "model/scaler.pkl"
scaler_y_path = "model/scaler_y.pkl"

# Load model and scalers
final_model = pickle.load(open(model_path, 'rb'))
scaler_X = pickle.load(open(scaler_X_path, 'rb'))
scaler_y = pickle.load(open(scaler_y_path, 'rb'))

# Streamlit App Title
st.title("🧠 Parkinson's Risk Prediction App")

st.write("""
### Enter the following medical parameters to predict the risk percentage:
""")

# User Input Fields
putamen_r = st.number_input("DATSCAN_PUTAMEN_R", min_value=0.0, format="%.2f")
putamen_l = st.number_input("DATSCAN_PUTAMEN_L", min_value=0.0, format="%.2f")
caudate_r = st.number_input("DATSCAN_CAUDATE_R", min_value=0.0, format="%.2f")
caudate_l = st.number_input("DATSCAN_CAUDATE_L", min_value=0.0, format="%.2f")
NP3TOT = st.number_input("NP3TOT (Motor Symptoms Score)", min_value=0, format="%d")
UPSIT_PRCNTGE = st.number_input("UPSIT_PRCNTGE (Smell Test Score)", min_value=0.0, format="%.2f")
COGCHG = st.selectbox("Cognitive Change (0 = No, 1 = Yes)", [0, 1])

# Make Prediction on Button Click
if st.button("Predict Parkinson's Risk"):
    try:
        # Arrange input data
        input_data = np.array([[putamen_r, putamen_l, caudate_r, caudate_l, NP3TOT, UPSIT_PRCNTGE, COGCHG]])

        # Apply feature scaling
        input_data_scaled = scaler_X.transform(input_data)

        # Make Prediction
        pred_scaled = final_model.predict(input_data_scaled)

        # Convert prediction back to original scale
        risk_percent = float(scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0])

        # Risk Status Classification
        risk_status = "No significant risk detected (Negative)" if risk_percent <= 20 else "Significant risk detected (Positive)"

        # Display Results
        st.success(f"🧠 **Predicted Parkinson's Risk Percentage:** {risk_percent:.2f}%")
        st.info(f"**Risk Status:** {risk_status}")

    except Exception as e:
        st.error(f"Error: {str(e)}")
