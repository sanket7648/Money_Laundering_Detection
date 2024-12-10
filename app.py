import streamlit as st
import pandas as pd
import pickle
import tensorflow as tf
import plotly.express as px
import matplotlib.pyplot as plt  # Import matplotlib for line chart

# Load models efficiently with caching
@st.cache_resource
def load_model(model_name):
    if model_name == 'Logistic Regression':
        return pickle.load(open('log_model.pkl', 'rb'))
    elif model_name == 'Linear Regression':
        return pickle.load(open('lin_model.pkl', 'rb'))
    elif model_name == 'KNN':
        return pickle.load(open('knn_model.pkl', 'rb'))
    elif model_name == 'Naive Bayes':
        return pickle.load(open('nb_model.pkl', 'rb'))
    elif model_name == 'Random Forest':
        return pickle.load(open('rf_model.pkl', 'rb'))
    elif model_name == 'SVM':
        return pickle.load(open('svm_model.pkl', 'rb'))
    elif model_name == 'XGBoost':
        return pickle.load(open('xgb_model.pkl', 'rb'))
    elif model_name == 'DNN':
        return tf.keras.models.load_model('dnn_model.h5')

# Use pandas to load data (for smaller datasets)
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

# Streamlit UI setup
st.title("Fraud Detection System")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Load and clean the data using pandas for faster processing
    data = load_data(uploaded_file)
    data.columns = data.columns.str.strip()

    if 'Class' not in data.columns:
        st.error("The dataset must contain a 'Class' column for the target variable.")
    else:
        X = data.drop('Class', axis=1)
        y = data['Class']

        # Allow user to select model for prediction
        model_choice = st.selectbox("Choose the model for prediction", 
                                    ['Logistic Regression', 'Linear Regression', 'KNN', 'Naive Bayes', 
                                     'Random Forest', 'SVM', 'XGBoost', 'DNN'])
        model = load_model(model_choice)

        # Predict in batch (avoiding row-by-row prediction for faster execution)
        predictions = model.predict(X.values)

        # Display prediction results
        st.write("Prediction Results (First 1000 predictions):")
        st.write(predictions[:1000])

        # Visualization - Bar plot with Plotly for faster rendering
        st.subheader("Prediction Results Chart - Bar Plot")
        fig = px.bar(x=predictions.flatten(), title="Prediction Results")
        st.plotly_chart(fig)

        # Remove Pie chart and replace with Line chart
        st.subheader("Prediction Results Chart - Line Chart")
        # Create a line chart displaying the predictions
        fig3 = plt.figure(figsize=(10, 6))
        plt.plot(predictions.flatten(), color='blue', marker='o', linestyle='-', markersize=3)
        plt.title('Prediction Results Over Index')
        plt.xlabel('Index')
        plt.ylabel('Prediction (Fraud = 1, Not Fraud = 0)')
        plt.grid(True)
        st.pyplot(fig3)

        # Visualization - Scatter plot with Plotly
        st.subheader("Prediction Results Chart - Scatter Plot")
        fig3 = px.scatter(x=range(len(predictions.flatten())), y=predictions.flatten(), 
                          labels={"x": "Index", "y": "Prediction"}, title="Prediction Scatter Plot")
        st.plotly_chart(fig3)

        # Allow user to enter new data for prediction
        st.subheader("Enter New Data for Prediction:")
        user_input = []
        for column in X.columns:
            user_input.append(st.number_input(f"Enter value for {column}:"))

        if st.button('Predict'):
            user_input = [user_input]
            prediction = model.predict(user_input)
            st.write(f"Prediction: {'Fraud' if prediction[0] == 1 else 'Not Fraud'}")
