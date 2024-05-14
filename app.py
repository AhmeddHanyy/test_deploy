import streamlit as st
import joblib
import numpy as np

# Load the trained model
try:
    model = joblib.load("hanyy.pkl")
except Exception as e:
    st.error("Error loading model: " + str(e))

# Define the input data model
class InputData:
    features: list

# Define the Streamlit app
def main():
    st.title("Prediction App")

    # Input features
    features = st.text_input("Enter features (comma-separated):")

    if st.button("Predict"):
        # Prepare input features for prediction
        input_features = np.array([float(x) for x in features.split(",")])

        # Make prediction
        prediction = model.predict(input_features.reshape(1, -1))

        # Display prediction
        st.success("Prediction: " + str(prediction[0]))

# Run the app
if __name__ == "__main__":
    main()
