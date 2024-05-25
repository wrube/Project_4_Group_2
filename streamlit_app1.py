import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sklearn

warnings.filterwarnings("ignore", category=UserWarning)

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def select_row(df, index):
    try:
        selected_row = df.iloc[[index]]
    except IndexError:
        st.error(f"Index {index} is out of bounds for the DataFrame.")
        return None
    return selected_row

def predict_row(loaded_model, row):
    x = row.drop(columns=["Is Fraud?"])
    prediction = loaded_model.predict(x)
    prediction_proba = loaded_model.predict_proba(x)
    pred_dict = x.iloc[0].to_dict()
    return pred_dict, prediction[0], prediction_proba[0]

def display_prediction(pred_dict, prediction, prediction_proba):
    st.write("**Prediction:**", prediction)
    st.write("**Prediction Probabilities:**", prediction_proba)
    st.write("**Details of the selected row:**")
    st.json(pred_dict)

def evaluate_model(loaded_model, df):
    X = df.drop(columns=["Is Fraud?"])
    y = df["Is Fraud?"]
    y_pred = loaded_model.predict(X)
    y_pred_proba = loaded_model.predict_proba(X)[:, 1]

    accuracy = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred)
    cm = confusion_matrix(y, y_pred)

    st.write("**Accuracy:**", accuracy)
    st.write("**Classification Report:**")
    st.text(report)

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    st.pyplot(plt)

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

def display_feature_importance(loaded_model, df, top_n=20):
    if hasattr(loaded_model, 'named_steps'):
        model = loaded_model.named_steps['classifier']
        preprocessor = loaded_model.named_steps['preprocessor']
    else:
        model = loaded_model
        preprocessor = None

    if preprocessor is not None:
        preprocessor.fit(df.drop(columns=["Is Fraud?"]))
        feature_names = get_feature_names(preprocessor)
    else:
        feature_names = [f"Feature {i}" for i in range(len(model.feature_importances_))]

    if hasattr(model, 'feature_importances_'):
        feature_importances = model.feature_importances_
        sorted_idx = np.argsort(feature_importances)[-top_n:]

        plt.figure(figsize=(10, 7))
        plt.barh(range(len(sorted_idx)), feature_importances[sorted_idx], align='center')
        plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importances')
        st.pyplot(plt)
    else:
        st.error("The model does not have feature_importances_ attribute.")

def main():
    st.title("Model Evaluation and Prediction")

    models = {
        '1': ('XGBoost', 'models/trained_model100_xgboost.joblib'),
        '2': ('Logistic Regression', 'models/trained_model100_logisticregression.joblib'),
        '3': ('Random Forest', 'models/trained_model100_randomforest.joblib')
    }

    st.sidebar.title("Select Model")
    model_choice = st.sidebar.selectbox("Choose a model to load:", options=list(models.keys()), format_func=lambda x: models[x][0])
    
    model_name, model_path = models[model_choice]
    
    try:
        loaded_model = joblib.load(model_path)
        st.sidebar.write(f"Loaded model: {model_name}")
    except Exception as e:
        st.sidebar.error(f"Failed to load model {model_name} from {model_path}. Error: {e}")
        return

    data_path = 'data/transactions_users_100.csv'
    df = load_data(data_path)

    option = st.sidebar.selectbox("Choose an action", ["Make a prediction", "Evaluate the model", "Display feature importance", "Exit"])

    if option == "Make a prediction":
        row_index = st.sidebar.number_input(f"Enter the row index (0 to {len(df) - 1}) for prediction:", min_value=0, max_value=len(df) - 1, step=1)
        row = select_row(df, row_index)
        if row is not None:
            pred_dict, prediction, prediction_proba = predict_row(loaded_model, row)
            display_prediction(pred_dict, prediction, prediction_proba)

    elif option == "Evaluate the model":
        evaluate_model(loaded_model, df)

    elif option == "Display feature importance":
        display_feature_importance(loaded_model, df, top_n=20)

    elif option == "Exit":
        st.stop()

if __name__ == "__main__":
    st.write("Using scikit-learn version:", sklearn.__version__)
    main()