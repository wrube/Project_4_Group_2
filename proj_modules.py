import streamlit as st
import numpy as np
import pandas as pd
import joblib
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