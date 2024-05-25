import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sklearn
import os


#----------------------------------------------------------
# Functions for data import
#----------------------------------------------------------


def add_leading_zero_to_zipcode(item):
    item_str = str(item)  # Ensure the item is a string
    item_str = item_str.replace('.0', '') # remove trailing '.0'
    
    if len(item_str) == 4:
        return '0' + item_str
    elif len(item_str) == 3:
        return '00' + item_str
    
    elif item_str == '10072': # catch bad New York Zipcode
        return '10001'
    elif item_str == '30399': # catch bad Atlanta Zipcode
        return '30303'
    elif item_str == '94101': # catch bad San Francisco Zipcode
        return '94102'
    elif item_str == '92164': # catch bad San Diego
        return '92101'
    elif item_str == '98205': # catch bad Everett WA
        return '98201'
    elif item_str == '29573':
        return '29574'
    elif item_str == '19388':
        return '19390'
    elif item_str == '19640': # Reading PA
        return '19601'
    elif item_str == '16532': # Erie PA
        return '16501'
    elif item_str == '14645': # Rochester NY
        return '14604'
    elif item_str == '19483':
        return '19481'      # Valley Forge PA
    elif item_str == '17767':
        return '17751'      # Salona PA
    elif item_str == '45418':
        return '45390'      # Dayton OH
    elif item_str == '30330':
        return '30329'      # Atlanta GA
    elif item_str == '25965': 
        return '25976'      # Elton WV
    
    return item_str

def remove_dollar_and_convert(item):
    # Remove the dollar sign and convert to integer
    return np.int32(item.replace('$', ''))

def remove_dollar_and_convert_float(item):
    # Remove the dollar sign and convert to float
    return np.float64(item.replace('$', ''))

def convert_yes_no_to_binary(item):
    item_lower = str(item).lower()
    if item_lower == 'yes':
        result = 1
    else:
        result = 0 
    return np.int8(result)

# Utility Functions
def load_data(file_path):
    """Load the dataset from a CSV file."""
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None
    return df

def select_row(df, index):
    """Select a specific row from the dataframe by index."""
    try:
        selected_row = df.iloc[[index]]
    except IndexError:
        st.error(f"Index {index} is out of bounds for the DataFrame.")
        return None
    return selected_row

def predict_row(loaded_model, row):
    """Predict the class and probability for a specific row."""
    try:
        x = row.drop(columns=["Is Fraud?"])
        prediction = loaded_model.predict(x)
        prediction_proba = loaded_model.predict_proba(x)
        pred_dict = x.iloc[0].to_dict()
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None, None
    return pred_dict, prediction[0], prediction_proba[0]

def display_prediction(pred_dict, prediction, prediction_proba):
    """Display the prediction results."""
    st.subheader("Prediction Result")
    st.markdown(f"**Prediction:** {'Fraudulent' if prediction == 1 else 'Legitimate'}")
    st.markdown(f"**Prediction Probabilities:** {prediction_proba}")
    
    st.markdown("**Details of the selected row:**")
    pred_df = pd.DataFrame(pred_dict.items(), columns=["Feature", "Value"])
    st.table(pred_df)

def evaluate_model(loaded_model, df):
    """Evaluate the model and display performance metrics."""
    try:
        X = df.drop(columns=["Is Fraud?"])
        y = df["Is Fraud?"]
        y_pred = loaded_model.predict(X)
        y_pred_proba = loaded_model.predict_proba(X)[:, 1]

        accuracy = accuracy_score(y, y_pred)
        
        # Convert classification report to DataFrame
        report_dict = classification_report(y, y_pred, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        
        cm = confusion_matrix(y, y_pred)

        st.write("**Accuracy:**", accuracy)
        st.write("**Classification Report:**")
        st.table(report_df)

        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        st.pyplot(plt)
    except Exception as e:
        st.error(f"Error during model evaluation: {e}")

def get_feature_names(column_transformer):
    """Get feature names from all transformers in the ColumnTransformer."""
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

def get_feature_names(column_transformer):
    """Get feature names from all transformers in the ColumnTransformer."""
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
    """Display the top N feature importances of the model."""
    try:
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
            sorted_idx = sorted_idx[::-1]  # Reverse to descending order

            fig, ax = plt.subplots(figsize=(10, 7))
            sns.barplot(x=feature_importances[sorted_idx], y=[feature_names[i] for i in sorted_idx], palette="viridis", ax=ax)
            ax.set_xlabel('Feature Importance')
            ax.set_ylabel('Features')
            ax.set_title(f'Top {top_n} Feature Importances')
            st.pyplot(fig)
        else:
            st.error("The model does not have feature_importances_ attribute.")
    except Exception as e:
        st.error(f"Error displaying feature importances: {e}")