# import modules

import streamlit as st
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, recall_score, precision_score
from imblearn.pipeline import Pipeline  
from imblearn.under_sampling import RandomUnderSampler


from proj_modules import *


# ----------------------------------------------------------------------------------------------------------------
# session state variables
# ----------------------------------------------------------------------------------------------------------------

# if 'final_cleaned_df' not in st.session_state:
#     st.session_state.final_cleaned_df = pd.DataFrame()

if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}

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
    
    nn_model_path = st.sidebar.text_input("Neural Network",
                                   value="models/trained_model100_neuralnetwork.joblib",
                                      )
    


    model_paths = {'log_reg': log_reg_model_path, 
                   'random_forest': random_forest_path, 
                   'XGBoost': xgboost_model_path,
                   'neural_network': nn_model_path}

    load_models_button = st.sidebar.button("Load Models",
                      )
    
    if load_models_button:
        load_pretrained_models(model_paths, st.session_state['trained_models'])



# ----------------------------------------------------------------------------------------------------------------
# Main Page
# ----------------------------------------------------------------------------------------------------------------

intro_tab, analysis_tab = st.tabs(["Initial Modelling and Tests",
                                   "Model Build and Analysis"])

with intro_tab:

    data_2 = {
        'model': ['Logistic Regression', 'Random Forest', 'XGBoost', 'Neural Network'],
        'accuracy': [0.85, 0.92, 0.92, 0.88],
        'precision_nf': [1.00, 1.00, 1.00, 1.0],
        'recall_nf': [0.85, 0.92, 0.92, 0.88],
        'f1_nf': [0.92, 0.96, 0.96, 0.94],
        'precision_f': [0.01, 0.01, 0.01, 0.01],
        'recall_f': [0.86, 0.90, 0.93, 0.88],
        'f1_f': [0.01, 0.02, 0.02, 0.01]
    }
    df_100 = pd.DataFrame(data_2)


    data_3 = {
        'model': ['Logistic Regression', 'Random Forest', 'XGBoost', 'Neural Network'],
        'accuracy': [0.93, 0.92, 0.94, 0.89],
        'precision_nf': [1.0, 1.0, 1.0, 1.0],
        'recall_nf': [0.93, 0.92, 0.94, 0.88],
        'f1_nf': [0.96, 0.96, 0.97, 0.94],
        'precision_f': [0.01, 0.01, 0.02, 0.01],
        'recall_f': [0.93, 0.94, 0.97, 0.98],
        'f1_f': [0.03, 0.03, 0.04, 0.02]
    }
    df_500 = pd.DataFrame(data_3)

    st.markdown("""
    # 🧪 Modelling

## What we tried:
Throughout the analysis, the following Machine Learning Models were used to assess the data:
1) K-Means (KM): Clusters data based on feature similarities
2) Logistic Regression (LR): Algorithm used for binary classification that models the probability of a binary outcome using a logistic function.
3) Random Forest (RF): Constructs multiple decision trees during training and outputs the mode of the classes (classification) or mean prediction (regression) of the individual trees.
4) XGBoost (XG): Builds models in a stage-wise fashion to correct errors made by previous models.
5) Deep Neural Network (DNN): A neural network with multiple hidden layers that can learn complex patterns in data through hierarchical feature learning.

## Phase 1- Initial tests:
After cleaning the data, the first step was to independently run all models to form a foundational base for future tests. No manipulation or model tweaking was conducted at this stage
The following information was gleaned:
- The KM model showed no useful relationship or information, likely due to the binary nature of the target column. No further testing required
- The CNN and LR models showed very little promise, with 100% true-negatives for non-fraud columns, and 100% false-negatives in the fraud columns
- The RF and XG models attained some measure of true-positive incidences of fraud, but capped at 50% for XGBoost
During analysis, it was concluded that the primary issue of the dataset was the great disparity in values between fraud and non-fraud, with a ratio of 1:270

## Phase 2- Model manipulation and standardisation
With a baseline established, the next step was to standardise each of the tests. To do this, a pipeline was created to process the data, and was adapted to fit the needs of all models. Furthermore, to handle the imbalance of outcomes, undersampling was implemented during the training phase. These models were all conducted using 100 of the available 2000 users, and the results are tabulated below.

    """
    )
    st.dataframe(df_100)

    st.markdown(""" 

At this stage, the following was observed:
- All models now showed an apptitude for classifying fraud, alligning significantly better with the aims set out in the project proposal. The precision and f1 valuese are low across the models due to the imbalance of testing data
- The weakest model was the LR, with the lowest values in all categories
- Second weakest model was the NN for similar reasons to LR
- With similar results, the only aspect that raised XG over RF was an extra 3% ability to classify true-positives

## Phase 3- Modeling on larger datasets
With success during phase 2, all models showed a relative amount of potential and so all models were conducted with a dataset of 450. The tabulated results are shown below.

""")

    st.dataframe(df_500)

    st.markdown(""" 

The final observations are as follows:
- All models showed some measure of improvement, however precision and f1 for fraud is still low due to the imbalance of test data
- LR: Increased scores for all measures, and was the model with the greatest improvement
- RF: Only improved in recalling fraud
- XG: Increased slightly in all measures
- NN: Only improved in recalling fraud, but to a high measure

## Conclusive outcome
After the modelling at 450 users, the overall best model can be argued depending on the overall goal of use:

- The model with the highest ability to predict fraud is the NN model, however the accuracy and recall of non-fraud values is the lowest of all models
- The model with the highest recall of both fraud and non-fraud collectively is the XG model. While lower by 1% recalling fraud than the NN model, the added accuracy and non-fraud model more than make up for it
- For the purpose of this showcase, we will be demonstrating with the XGBoost model
""")

with analysis_tab:



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

            build_log_reg, build_random_forest, build_xgboost, build_neural_network = st.tabs([
                "Logistic Regression", 
                "Random Forest",
                "XGBoost",
                "Neural Network"
            ])

            # declare alias for trained models
            ss_models = st.session_state.trained_models

            with build_log_reg:

                # ----------------------------------------------------------------------------------------------------------------
                # Build Logistic Regression
                # ----------------------------------------------------------------------------------------------------------------
                col1, col2 = st.columns(2)
                with col1:
                    # Define the model
                    ss_key = 'log_reg'
                    max_iterations = st.number_input("Maximum Iterations", min_value=1, value=1000)
                    
                    lr_model = set_logistic_regression_model(max_iterations)

                    # Hyperparameter tuning using GridSearchCV
                    st.write('Use an grid parameter optimiser')
                    with st.echo():
                        lr_param_grid = {
                                'classifier__C': [0.1, 1.0, 10.0],
                                'classifier__penalty': ['l1', 'l2'],
                                'classifier__solver': ['liblinear', 'saga']
                                }
                with col2:
                    optimise_model(X_train, y_train, lr_model, preprocessor, lr_param_grid, ss_models, ss_key)


            
            with build_random_forest:

                # ----------------------------------------------------------------------------------------------------------------
                # Build Random Forest
                # ----------------------------------------------------------------------------------------------------------------
                col1, col2 = st.columns(2)
                with col1:
                    # Define the model
                    ss_key = 'random_forest'
                    rf_model = set_random_forest_model()

                    # Hyperparameter tuning using GridSearchCV
                    st.write('Use an grid parameter optimiser')
                    with st.echo():
                        rf_param_grid = {
                            'classifier__n_estimators': [100, 200],
                            'classifier__max_depth': [None, 10, 20],
                            'classifier__min_samples_split': [2, 5],
                            'classifier__min_samples_leaf': [1, 2]
                            }
                with col2:
                    optimise_model(X_train, y_train, rf_model, preprocessor, rf_param_grid, ss_models, ss_key)

            with build_xgboost:

                # ----------------------------------------------------------------------------------------------------------------
                # Build XGBoost Model
                # ----------------------------------------------------------------------------------------------------------------
                col1, col2 = st.columns(2)
                with col1:
                    # Define the model
                    ss_key = 'XGBoost'
                    xgb_model = set_xgboost_model()

                    # Hyperparameter tuning using GridSearchCV
                    st.write('Use an grid parameter optimiser')
                    with st.echo():
                        xgb_param_grid = {
                            'classifier__n_estimators': [100, 200],
                            'classifier__max_depth': [3, 6, 10],
                            'classifier__learning_rate': [0.01, 0.1, 0.2],
                            'classifier__subsample': [0.8, 1.0],
                            'classifier__colsample_bytree': [0.8, 1.0]
                        }
                with col2:
                    optimise_model(X_train, y_train, xgb_model, preprocessor, xgb_param_grid, ss_models, ss_key)


            with build_neural_network:

                # ----------------------------------------------------------------------------------------------------------------
                # Build Neural Network Model
                # ----------------------------------------------------------------------------------------------------------------
                col1, col2 = st.columns(2)
                with col1:
                    # Define the model
                    ss_key = 'neural_network'
                    nn_model = set_nn_model()

                    # Hyperparameter tuning using GridSearchCV
                    st.write('Use an grid parameter optimiser')
                    with st.echo():
                        nn_param_grid = {
                             'classifier__hidden_layer_sizes': [(50,), (100,), (50, 50)],
                             'classifier__activation': ['relu', 'tanh'],
                             'classifier__solver': ['adam', 'sgd'],
                             'classifier__alpha': [0.0001, 0.001],
                             'classifier__learning_rate': ['constant', 'adaptive'] 
                        }
                with col2:
                    optimise_model(X_train, y_train, nn_model, preprocessor, nn_param_grid, ss_models, ss_key)




        # ----------------------------------------------------------------------------------------------------------------
        # Set the Tabs For Displaying Results
        # ----------------------------------------------------------------------------------------------------------------
        st.header("Modelling Outputs on Test Set")

        st.markdown(f"""This test set has {y_test.value_counts()[1]} 'Fraudulent' cases.""")

        # set the tabs
        tab_log_reg, tab_random_forest, tab_xgboost, tab_nn = st.tabs(["Logistic Regression Model", 
                                                            "Random Forest Model", 
                                                            "XGBoost Model",
                                                            "Neural Network"]
                                                            )
        with tab_log_reg:
            # ----------------------------------------------------------------------------------------------------------------
            # Logistic Regression Results
            # ----------------------------------------------------------------------------------------------------------------
            rf_model = 'log_reg'
            if rf_model in st.session_state.trained_models:            

                ml_model = st.session_state.trained_models[rf_model]

                display_model_report(ml_model, X_test, y_test)

        with tab_random_forest:
            # ----------------------------------------------------------------------------------------------------------------
            # Random Forest Results
            # ----------------------------------------------------------------------------------------------------------------
            rf_model = 'random_forest'
            if rf_model in st.session_state.trained_models:            

                ml_model = st.session_state.trained_models[rf_model]

                display_model_report(ml_model, X_test, y_test)

        with tab_xgboost:
            # ----------------------------------------------------------------------------------------------------------------
            # Logistic Regression Results
            # ----------------------------------------------------------------------------------------------------------------
            rf_model = 'XGBoost'
            if rf_model in st.session_state.trained_models:            

                ml_model = st.session_state.trained_models[rf_model]

                display_model_report(ml_model, X_test, y_test)


        with tab_nn:
            # ----------------------------------------------------------------------------------------------------------------
            # Neural Network Results
            # ----------------------------------------------------------------------------------------------------------------
            nn_model = 'neural_network'
            if nn_model in st.session_state.trained_models:            

                ml_model = st.session_state.trained_models[nn_model]

                display_model_report(ml_model, X_test, y_test)

        # ----------------------------------------------------------------------------------------------------------------
        # Neural Network Results
        # ----------------------------------------------------------------------------------------------------------------
        compare_models_button = st.button("Compare Model Metrics")

        if compare_models_button:
            accuracies = {}
            precisions = {}
            recalls = {}
            
            for model_name, model in st.session_state.trained_models.items():
                y_pred = model.predict(X_test)
                accuracies[model_name] = accuracy_score(y_test, y_pred)
                precisions[model_name] = precision_score(y_test, y_pred)
                recalls[model_name] = recall_score(y_test, y_pred)
            metrics_df = pd.DataFrame([accuracies, precisions, recalls])
            metrics_df = metrics_df.rename(index={0: 'Accuracy',
                                                  1: 'Precision',
                                                  2: 'Recall'}).T

            # st.dataframe(metrics_df)
            n_metrics = len(metrics_df)

            x_axis = np.arange(n_metrics) + 1
            labels = metrics_df.index

            f, (ax1, ax2, ax3) = plt.subplots(1, 3)

            ax1.bar(x_axis, metrics_df['Accuracy'])
            ax1.set_xticks(x_axis, labels, rotation=60)
            ax1.set_title('Accuracy')
            ax1.set_ylabel('Fraction')
            
            ax2.bar(x_axis, metrics_df['Precision'])
            ax2.set_xticks(x_axis, labels, rotation=60)
            ax2.set_title('Precision')
            ax2.set_ylabel('Fraction')

            ax3.bar(x_axis, metrics_df['Recall'])
            ax3.set_xticks(x_axis, labels, rotation=60)
            ax3.set_title('Recall')
            ax3.set_ylabel('Fraction')
            
            f.set_figheight(3)
            f.tight_layout(pad=0.4, w_pad=0.5)
            st.pyplot(f)

