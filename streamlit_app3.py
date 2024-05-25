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

# Utility Functions
def load_data(file):
    """Load the dataset from a file."""
    try:
        df = pd.read_csv(file)
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
    prediction_text = "Legitimate" if prediction == 0 else "Fraudulent"
    prediction_color = "green" if prediction == 0 else "red"
    
    st.markdown(f"<span style='color:{prediction_color}; font-weight:bold;'>Prediction: {prediction_text}</span>", unsafe_allow_html=True)
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

        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Legitimate', 'Fraudulent'], yticklabels=['Legitimate', 'Fraudulent'], ax=ax)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        st.pyplot(fig)
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

def display_summary():
    """Display a summary of evaluation metrics in the context of credit card fraud detection."""
    st.title("Evaluation Metrics Summary in Credit Card Fraud Detection")

    st.header("Precision")
    st.markdown("""
    **Definition:** Precision is the ratio of true positive predictions to the total predicted positives.
    
    **In terms of credit card fraud:** Precision measures how many of the transactions flagged as fraudulent are actually fraudulent. High precision indicates a low number of false positives (legitimate transactions incorrectly flagged as fraudulent).
    """)

    st.header("Recall")
    st.markdown("""
    **Definition:** Recall is the ratio of true positive predictions to the total actual positives.
    
    **In terms of credit card fraud:** Recall measures how many of the actual fraudulent transactions are correctly identified by the model. High recall indicates a low number of false negatives (fraudulent transactions that are missed by the model).
    """)

    st.header("F1-Score")
    st.markdown("""
    **Definition:** The F1-score is the harmonic mean of precision and recall, providing a single metric that balances both concerns.
    
    **In terms of credit card fraud:** The F1-score provides a balance between precision and recall, helping to ensure that the model does not favor one over the other. This is crucial in fraud detection, where both false positives and false negatives have significant costs.
    """)

    st.header("Accuracy")
    st.markdown("""
    **Definition:** Accuracy is the ratio of correct predictions (both true positives and true negatives) to the total number of predictions.
    
    **In terms of credit card fraud:** Accuracy measures the overall correctness of the model's predictions. However, in fraud detection, accuracy can be misleading if the dataset is highly imbalanced (i.e., there are many more legitimate transactions than fraudulent ones), as a model could achieve high accuracy by simply predicting all transactions as legitimate.
    """)

    st.header("False Negatives")
    st.markdown("""
    **Definition:** A false negative occurs when a fraudulent transaction is incorrectly classified as legitimate.
    
    **Impact:**
    - Financial Loss: Undetected fraudulent transactions lead to direct financial losses for the credit card company and the cardholder.
    - Customer Trust: If a fraudulent transaction goes unnoticed, it can severely damage customer trust and satisfaction.
    - Security Risks: Allowing fraudulent activities to proceed can encourage more sophisticated fraud attempts in the future.
    """)

    st.header("False Positives")
    st.markdown("""
    **Definition:** A false positive occurs when a legitimate transaction is incorrectly classified as fraudulent.
    
    **Impact:**
    - Customer Inconvenience: Legitimate cardholders may face transaction declines, leading to frustration and inconvenience.
    - Operational Costs: Handling false positives requires additional resources for customer service and fraud investigation teams.
    - Customer Experience: Frequent false positives can lead to poor customer experiences, potentially resulting in customers switching to competitors.
    """)

    st.header("Which is More Acceptable?")
    st.markdown("""
    **In credit card fraud detection, false positives are generally more acceptable than false negatives for the following reasons:**
    1. Financial Impact: The direct financial loss from false negatives (missed fraud) is typically more significant than the operational costs of handling false positives.
    2. Customer Trust: Protecting customers from fraud (even at the cost of some inconvenience) helps maintain trust and confidence in the credit card company.
    3. Security: Proactively catching potential fraud helps in maintaining the overall security of the system and deterring fraudsters.
    
    **Balancing Both:** While minimizing false negatives is crucial, an effective fraud detection system aims to strike a balance where both false positives and false negatives are kept at acceptable levels. This balance is typically achieved by optimizing the model for metrics like the F1-score, which considers both precision (minimizing false positives) and recall (minimizing false negatives).
    """)

    st.header("Understanding Probability in the System")
    st.markdown("""
    When a machine learning model predicts whether a transaction is fraudulent or not, it often outputs a probability score. This score represents the likelihood that the transaction belongs to a particular class (fraudulent or legitimate).

    **How Probabilities are Used:**
    1. Threshold-Based Classification:
        - The model outputs a probability between 0 and 1.
        - A threshold is set (commonly 0.5). If the probability is above the threshold, the transaction is classified as fraudulent; otherwise, it is classified as legitimate.
    2. Confidence in Predictions:
        - The probability score indicates the model's confidence in its prediction.
        - For example, a probability of 0.9 for a transaction being fraudulent means the model is 90% confident in its prediction.
    3. Decision Making:
        - Probabilities can be used to prioritize investigations. Higher probability transactions can be reviewed first.
        - Adjusting the threshold can balance between false positives and false negatives based on business needs.
    
    **Practical Implications:**
    1. High Probability of Fraud (e.g., >0.8):
        - The system is highly confident that the transaction is fraudulent.
        - Immediate action can be taken, such as blocking the transaction or alerting the cardholder.
    2. Low Probability of Fraud (e.g., <0.2):
        - The system is highly confident that the transaction is legitimate.
        - The transaction can be processed without additional checks.
    3. Intermediate Probability (e.g., 0.4 to 0.6):
        - There is uncertainty in the prediction.
        - These transactions might be flagged for further review by fraud analysts.
    
    **Example Explanation:**
    Consider a transaction where the model outputs a probability of 0.85 for being fraudulent:
    - System Action: The system will likely classify this transaction as fraudulent (assuming a threshold of 0.5) and may trigger an alert or block the transaction.
    - Confidence Level: The model is 85% confident in this prediction. This high probability suggests a strong likelihood of fraud.
    """)

    st.header("Feature Importance")
    st.markdown("""
    **Definition:** Feature importance refers to techniques that assign a score to input features based on how useful they are at predicting a target variable.

    **In terms of credit card fraud:** Feature importance helps in identifying which features (such as transaction amount, location, time, etc.) have the most influence on predicting whether a transaction is fraudulent or legitimate.

    **How it Works:**
    - **Tree-based Models:** For models like Random Forest or XGBoost, feature importance can be derived from the model itself as these models can compute the importance of each feature.
    - **Permutation Importance:** This technique involves shuffling the values of each feature and measuring how much the shuffling decreases the model's performance. Features that are important will see a significant decrease in performance when their values are shuffled.

    **Why it Matters:**
    - **Model Interpretation:** Understanding which features are important helps in interpreting the model's decisions, making it easier to trust and verify the model's predictions.
    - **Feature Selection:** By identifying the most important features, you can potentially reduce the number of features in the dataset, leading to simpler and more efficient models.
    - **Improving Model Performance:** Focusing on the most important features can help in improving the model's performance.

    **Example Explanation:**
    If the model indicates that the 'transaction amount' is the most important feature, it means that the transaction amount plays a significant role in determining whether a transaction is fraudulent or not. This insight can be valuable for developing targeted fraud detection strategies.
    """)

# Main Function
def main():
    st.title("Fraud Detection Dashboard")

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

    st.sidebar.title("Load Data")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

    # Use default data if no file is uploaded
    if uploaded_file is not None:
        df = load_data(uploaded_file)
    else:
        df = load_data('data/transactions_users_100.csv')
        st.sidebar.write("Using default dataset for analysis.")

    option = st.sidebar.selectbox("Choose an action", ["Make a prediction", "Evaluate the model", "Display feature importance", "Summary of Metrics", "Exit"])

    if option == "Make a prediction":
        row_index = st.sidebar.number_input(f"Enter the row index (0 to {len(df) - 1}) for prediction:", min_value=0, max_value=len(df) - 1, step=1)
        row = select_row(df, row_index)
        if row is not None:
            pred_dict, prediction, prediction_proba = predict_row(loaded_model, row)
            if pred_dict is not None:
                display_prediction(pred_dict, prediction, prediction_proba)
        st.markdown("This page allows you to make a prediction on a specific transaction by entering the row index or inputting values manually. You can view the prediction result and the details of the selected row.")
        with st.expander("Learn more"):
            st.markdown("""
                On this page, you can:
                - **Make a prediction**: Enter the row index of the transaction you want to predict.
                - **View prediction results**: See whether the transaction is predicted to be fraudulent or legitimate.
                - **View prediction probabilities**: Understand the confidence of the model's prediction.
                - **Details of the selected row**: Examine the features of the selected transaction.
            """)

    elif option == "Evaluate the model":
        evaluate_model(loaded_model, df)
        st.markdown("This page provides an evaluation of the model's performance on the entire dataset, including metrics such as accuracy, classification report, and confusion matrix.")
        with st.expander("Learn more"):
            st.markdown("""
                On this page, you can:
                - **Evaluate the model**: Assess the performance of the model using metrics like accuracy, precision, recall, and F1-score.
                - **View the confusion matrix**: Understand how many transactions were correctly and incorrectly classified.
            """)

    elif option == "Display feature importance":
        display_feature_importance(loaded_model, df, top_n=20)
        st.markdown("This page displays the top N feature importances used by the model. It helps in understanding which features are most influential in predicting the outcome.")
        with st.expander("Learn more"):
            st.markdown("""
                On this page, you can:
                - **View feature importances**: See which features have the most influence on the model's predictions.
                - **Understand feature importance**: Learn how each feature contributes to the model's decision-making process.
            """)

    elif option == "Summary of Metrics":
        display_summary()

    elif option == "Exit":
        st.markdown("You have chosen to exit the application. Thank you for using the model evaluation and prediction tool.")
        st.stop()

if __name__ == "__main__":
    st.write("Using scikit-learn version:", sklearn.__version__)
    main()