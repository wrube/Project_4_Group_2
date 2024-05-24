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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.pipeline import Pipeline  
from imblearn.under_sampling import RandomUnderSampler


#import modules
# sys.path.append(Path.cwd().parent)
from proj_modules import *


# ----------------------------------------------------------------------------------------------------------------
# session state variables
# ----------------------------------------------------------------------------------------------------------------

if 'final_cleaned_df' not in st.session_state:
    st.session_state.final_cleaned_df = pd.DataFrame()

if 'pre_trained_models' not in st.session_state:
    st.session_state.pre_trained_models = {}


# ****************************************************************************************************************
# ----------------------------------------------------------------------------------------------------------------
# setup page
# ----------------------------------------------------------------------------------------------------------------

st.set_page_config(page_title="Modelling", 
                   page_icon="🧪",
                   layout='wide')

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
# Modelling
            
## Initial Modelling 
            
 
            

"""
)

