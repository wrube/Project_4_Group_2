import pandas as pd
import numpy as np
import joblib
import streamlit as st
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and data
model_path = 'trained_model100_xgboost.joblib'
data_path = 'data/final_merged_100users.csv'
loaded_model = joblib.load(model_path)
df = pd.read_csv(data_path)

def get_feature_names(column_transformer):
    output_features = []
    for name, transformer, features in column_transformer.transformers_:
        if transformer == 'drop':
            continue
        if hasattr(transformer, 'get_feature_names_out'):
            output_features.extend(transformer.get_feature_names_out(features))
        elif hasattr(transformer, 'get_feature_names'):
            output_features.extend(transformer.get_feature_names())
        else:
            output_features.extend(features)
    return output_features

def select_row(index):
    try:
        selected_row = df.iloc[[index]]
    except IndexError:
        selected_row = None
    return selected_row

def predict_row(row):
    x = row.drop(columns=["Is Fraud?"])
    prediction = loaded_model.predict(x)
    prediction_proba = loaded_model.predict_proba(x)
    pred_dict = x.iloc[0].to_dict()
    return pred_dict, prediction[0], prediction_proba[0]

def evaluate_model():
    X = df.drop(columns=["Is Fraud?"])
    y = df["Is Fraud?"]
    y_pred = loaded_model.predict(X)
    y_pred_proba = loaded_model.predict_proba(X)[:, 1]
    accuracy = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    return accuracy, report, cm

def plot_confusion_matrix(cm):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    st.pyplot(plt)

# Streamlit application
st.title("Fraud Detection Prediction System")

# Sidebar for navigation
st.sidebar.title("Navigation")
option = st.sidebar.selectbox("Choose an option", ("Home", "Predict", "Evaluate"))

if option == "Home":
    st.write("Welcome to the Fraud Detection Prediction System!")
    st.write("Use the sidebar to navigate to different sections of the app.")

elif option == "Predict":
    st.header("Make a Prediction")
    row_index = st.number_input("Enter Row Index for Prediction", min_value=0, max_value=len(df)-1, value=0)
    if st.button("Predict"):
        row = select_row(row_index)
        if row is not None:
            pred_dict, prediction, prediction_proba = predict_row(row)
            st.write("### Prediction Result")
            st.write(f"Prediction: {prediction}")
            st.write(f"Prediction Probabilities: {prediction_proba}")
            st.write("### Input Features")
            st.json(pred_dict)
        else:
            st.error("Index out of bounds")

elif option == "Evaluate":
    st.header("Evaluate Model")
    if st.button("Evaluate"):
        accuracy, report, cm = evaluate_model()
        st.write("### Model Accuracy")
        st.write(accuracy)
        st.write("### Classification Report")
        st.text(report)
        st.write("### Confusion Matrix")
        plot_confusion_matrix(cm)