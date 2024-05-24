# import modules
import sys
import streamlit as st
import time
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from imblearn.pipeline import Pipeline  
from imblearn.under_sampling import RandomUnderSampler

import warnings


#import modules
# sys.path.append(Path.cwd().parent)
from proj_modules import *


# ----------------------------------------------------------------------------------------------------------------
# session state variables
# ----------------------------------------------------------------------------------------------------------------

# if 'final_cleaned_df' not in st.session_state:
#     st.session_state.final_cleaned_df = pd.DataFrame()

if 'pre_trained_models' not in st.session_state:
    st.session_state.pre_trained_models = {}

if 'training_features_df' not in st.session_state:
    st.session_state.training_features_df = pd.DataFrame()

if 'training_target_df' not in st.session_state:
    st.session_state.training_target_df = pd.DataFrame()

if 'test_features_df' not in st.session_state:
    st.session_state.test_features_df = pd.DataFrame()

if 'test_target_df' not in st.session_state:
    st.session_state.test_target_df = pd.DataFrame()


# ****************************************************************************************************************
# ----------------------------------------------------------------------------------------------------------------
# setup page
# ----------------------------------------------------------------------------------------------------------------

st.set_page_config(page_title="Modelling", 
                   page_icon="🧪",
                   layout='wide',
                   )

# ----------------------------------------------------------------------------------------------------------------
# sidebar setup
# ----------------------------------------------------------------------------------------------------------------
page_option = st.sidebar.radio("Choose:",
                 options=["Load Pre-trained models",
                            "Create Your Own Model"],
                 )

if page_option == "Load Pre-trained models":
    st.sidebar.markdown("#### Load pre-trained models")

    log_reg_model_path = st.sidebar.text_input("Logistic Regression",
                                   value="models/trained_model100_logisticregression.joblib",
                                      )
    random_forest_path = st.sidebar.text_input("Random Forest",
                                   value="models/trained_model100_randomforest.joblib",
                                      )
    xgboost_model_path = st.sidebar.text_input("XGBoost",
                                   value="models/trained_model100_xgboost.joblib",
                                      )
    
    model_paths = {'log_reg': log_reg_model_path, 
                   'random_forest': random_forest_path, 
                   'XGBoost': xgboost_model_path}

    load_models_button = st.sidebar.button("Load Models",
                      )
    
    if load_models_button:
        load_pretrained_models(model_paths, st.session_state['pre_trained_models'])


# ----------------------------------------------------------------------------------------------------------------
# Main Page
# ----------------------------------------------------------------------------------------------------------------


st.markdown("""
# 🧪 Modelling
            
## Initial Modelling and Tests
### What we tried

### What we learnt        
 
"""
)

if len(st.session_state.training_features_df) > 0:

    X_train = st.session_state.training_features_df
    X_test = st.session_state.test_features_df
    y_train = st.session_state.training_target_df.cat.codes
    y_test = st.session_state.test_target_df.cat.codes


    if page_option == "Create Your Own Model":
        st.header("Build Your Own Model")
        # ----------------------------------------------------------------------------------------------------------------
        # Preprocess Data with Pipelines
        # ----------------------------------------------------------------------------------------------------------------
        categorical_features = ['Use Chip',
                    'Merchant State',
                    'Errors?',
                    'Has Chip',
                    'International',
                    'Online',
                    'day_of_week',
                    'time_of_day']
        
        numerical_features = ['Amount',
                    'Per Capita Income - Zipcode',
                    'Yearly Income - Person',
                    'Total Debt',
                    'FICO Score',
                    'Num Credit Cards',
                    'Cards Issued',
                    'Age_at_transaction',
                    'income_to_debt',
                    'timestamp',
                    'distances']
        

        
        # Preprocessing for numerical data
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Preprocessing for categorical data
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Bundle preprocessing for numerical and categorical data
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # ----------------------------------------------------------------------------------------------------------------
        # Set the Tabs For Training Models
        # ----------------------------------------------------------------------------------------------------------------
        undersampling = st.toggle("Undersampling to handle imbalanced target?", value=True)
        build_log_reg, build_random_forest, build_xgboost, build_nn = st.tabs([
            "Logistic Regression", 
            "Random Forest",
            "XGBoost",
            "Neural Network"
        ])
        
        with build_log_reg:
            # Define the model
            max_iterations = st.number_input("Maximum Iterations", min_value=1, value=1000)
            # undersampling = st.toggle("Undersampling to handle imbalanced target?", value=True)
            model = LogisticRegression(random_state=42, max_iter=max_iterations)

            # Create the pipeline with RandomUnderSampler and preprocessing
            pipeline = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('undersampler', RandomUnderSampler(random_state=42)),
                    ('classifier', model)
                    ])
            # Hyperparameter tuning using GridSearchCV
            param_grid = {
                    'classifier__C': [0.1, 1.0, 10.0],
                    'classifier__penalty': ['l1', 'l2'],
                    'classifier__solver': ['liblinear', 'saga']
                    }
            
            warnings.filterwarnings("ignore")
            
            # Perform grid search with the pipeline, reduced n_jobs, and error_score='raise' for detailed errors
            grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=1, error_score='raise')

            fit_button = st.button("Fit the Model")
            if fit_button:
                grid_search.fit(X_train, y_train)

                # Best parameters and estimator
                best_params = grid_search.best_params_
                best_estimator = grid_search.best_estimator_

                st.write("\nBest Parameters:")
                st.write(best_params)






    # ----------------------------------------------------------------------------------------------------------------
    # Set the Tabs For Displaying Results
    # ----------------------------------------------------------------------------------------------------------------
    st.header("Modelling Outputs")
    # set the tabs
    tab_log_reg, tab_random_forest, tab_xgboost = st.tabs(["Logistic Regression Model", 
                                                        "Random Forest Model", 
                                                        "XGBoost Model"])
    with tab_log_reg:
        # ----------------------------------------------------------------------------------------------------------------
        # Logistic Regression Results
        # ----------------------------------------------------------------------------------------------------------------
        model = 'log_reg'
        if model in st.session_state.pre_trained_models:            

            ml_model = st.session_state.pre_trained_models[model]

            display_model_report(ml_model, X_test, y_test)

    with tab_random_forest:
        # ----------------------------------------------------------------------------------------------------------------
        # Random Forest Results
        # ----------------------------------------------------------------------------------------------------------------
        model = 'random_forest'
        if model in st.session_state.pre_trained_models:            

            ml_model = st.session_state.pre_trained_models[model]

            display_model_report(ml_model, X_test, y_test)

    with tab_xgboost:
        # ----------------------------------------------------------------------------------------------------------------
        # Logistic Regression Results
        # ----------------------------------------------------------------------------------------------------------------
        model = 'XGBoost'
        if model in st.session_state.pre_trained_models:            

            ml_model = st.session_state.pre_trained_models[model]

            display_model_report(ml_model, X_test, y_test)


