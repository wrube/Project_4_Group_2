import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, ConfusionMatrixDisplay
from imblearn.pipeline import Pipeline  
from imblearn.under_sampling import RandomUnderSampler
from pathlib import Path

import warnings

# ----------------------------------------------------------------------------------------------------------------
# session state variables
# ----------------------------------------------------------------------------------------------------------------

# if 'final_cleaned_df' not in st.session_state:
#     st.session_state.final_cleaned_df = pd.DataFrame()

# if 'pre_trained_models' not in st.session_state:
#     st.session_state.pre_trained_models = {}


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


# @st.cache_data
# def load_csv_data(columns, converters, dtypes):
#     df = pd.read_csv(users_path, 
#                         usecols=columns,
#                         converters=converters,
#                         dtype=dtypes
#                         )
#     return df

@st.cache_data
def load_csv_data(file, columns, _converters, _dtypes):
    df = pd.read_csv(file, 
                        usecols=columns,
                        converters=_converters,
                        dtype=_dtypes
                        )
    return df


def load_pretrained_models(path_dict, session_variable):
    print("Hi", type(session_variable))
    try:
        for model, path in path_dict.items():
            # Load the model
            loaded_model = load_model(path)
            print("Model loaded successfully")
            session_variable.update({model: loaded_model})
        st.sidebar.write("Models loaded successfully!")
    except Exception as e:
        st.sidebar.write(f"Error: Check models or model paths for {model}. {e}")

@st.cache_resource
def load_model(model_path):
    # Load the model
    loaded_model = joblib.load(model_path)
    return loaded_model


def display_model_report(model, X_test, y_test):

    # Predict using the loaded model
    y_pred = model.predict(X_test)

    col1, col2, col3 = st.columns([0.35, 0.25, 0.4])

    with col1:
    # Evaluation
        col1.subheader("Classification Report")

        st.dataframe(
            pd.DataFrame(classification_report(y_test, 
                        y_pred, 
                        target_names=['Not Fraud', 'Fraud'],
                        output_dict=True)).transpose()
                    )
        col_accuracy, col_recall = st.columns(2)
        with col_accuracy:
            st.markdown("#### Accuracy Score") 
            st.markdown(f"###### {accuracy_score(y_test, y_pred):.3}")
        with col_recall:
            st.markdown("#### Recall Score") 
            st.markdown(f"###### {recall_score(y_test, y_pred):.3}")

    with col2:
        col2.subheader("Confusion Matrix")                
        cm = confusion_matrix(y_test, y_pred)

        f, ax = plt.subplots(figsize=(2,2))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)

        disp.plot(ax=ax)
        disp.im_.colorbar.remove()


        st.pyplot(f)
    with col3:
        col3.subheader("Definitions for Fraudulent Metrics")
        st.markdown("""
- **Accuracy** is the percentage of examples correctly classified: $\dfrac{TP + TN}{P + N}$
- **Precision** is the percentage of predicted positives that were correctly classified:  $\dfrac{TP}{TP + FP}$
- **Recall** is the percentage of actual positives that were correctly classified: $\dfrac{TP}{TP + FN}$
        """
        )


@st.cache_data
def train_model(X_train, y_train, _grid_search):

    _grid_search.fit(X_train, y_train)



@st.cache_resource
def set_logistic_regression_model(max_iterations):  
    return LogisticRegression(random_state=42, max_iter=max_iterations)



@st.cache_resource
def set_random_forest_model():  
    return RandomForestClassifier(random_state=42)



@st.cache_resource
def set_xgboost_model():  
    return XGBClassifier(random_state=42, 
                         use_label_encoder=False, 
                         eval_metric='logloss')

@st.cache_resource
def set_nn_model():  
    return MLPClassifier(random_state=42, 
                         max_iter=500,
                         )

@st.cache_resource
def set_pipeline(_model, _preprocessor):
    pipeline = Pipeline(steps=[
        ('preprocessor', _preprocessor),
        ('undersampler', RandomUnderSampler(random_state=42)),
        ('classifier', _model)
        ])
    return pipeline

@st.cache_resource
def set_grid_search(_pipeline, _param_grid):
    grid_search = GridSearchCV(_pipeline, _param_grid, cv=5, scoring='accuracy', n_jobs=1, error_score='raise')
    return grid_search



def optimise_model(X_train, y_train, _model, _preprocessor, _param_grid, _ss_model, ss_key):
    warnings.filterwarnings("ignore")
    set_pipeline.clear()
    # Create the pipeline with RandomUnderSampler and preprocessing
    pipeline = set_pipeline(_model, _preprocessor)
            
    # Perform grid search with the pipeline, reduced n_jobs, and error_score='raise' for detailed errors
    set_grid_search.clear()
    grid_search = set_grid_search(pipeline, _param_grid)
    

    # create the fit model button
    fit_button = st.button("Optimise model parameters",
                           key=f"{ss_key}_optimise")
    if fit_button:
        
        
        train_model.clear()
        
        
        train_model(X_train, y_train, grid_search)


                # Best parameters and estimator
        best_params = grid_search.best_params_
        best_estimator = grid_search.best_estimator_

                # save to session state
        _ss_model[ss_key] = best_estimator

        st.write("\nBest Parameters:")
        st.write(best_params)

    # create the save model button
    save_name = st.text_input("Model Name (will be saved in the project 'models' folder)", 
                                value=f"{ss_key}_training_model.joblib",
                                key=f"{ss_key}_save_name")
    save_model_button = st.button("Save the model",
                                key=f"{ss_key}_save_button",
                                disabled=ss_key not in _ss_model.keys())
    
    # action save click
    if (save_model_button) & (ss_key in _ss_model.keys()):
        joblib.dump(_ss_model[ss_key], Path.cwd() / 'models' / save_name)
    


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