# import modules
import sys
import streamlit as st
import time
import numpy as np
from pathlib import Path
import pandas as pd


# import modules
sys.path.append(Path.cwd().parent)
from proj_modules import *

# ----------------------------------------------------------------------------------------------------------------
# session state variables
# ----------------------------------------------------------------------------------------------------------------

if 'users_csv_loaded' not in st.session_state:
    st.session_state.users_csv_loaded = False

if 'cards_csv_loaded' not in st.session_state:
    st.session_state.cards_csv_loaded = False

if 'transactions_csv_loaded' not in st.session_state:
    st.session_state.transactions_csv_loaded = False

if 'users_df' not in st.session_state:
    st.session_state.users_df = pd.DataFrame()

if 'cards_df' not in st.session_state:
    st.session_state.cards_df = pd.DataFrame()

if 'transactions_df' not in st.session_state:
    st.session_state.transactions_df = pd.DataFrame()


# ****************************************************************************************************************
# ----------------------------------------------------------------------------------------------------------------
# setup page
# ----------------------------------------------------------------------------------------------------------------


st.set_page_config(page_title="Raw Data from Kaggle", 
                   page_icon="ℹ️")

# ----------------------------------------------------------------------------------------------------------------
# page content
# ----------------------------------------------------------------------------------------------------------------

st.markdown("# Data Load")
st.sidebar.header("Data Load")

# ----------------------------------------------------------------------------------------------------------------
# Section Introduction
# ----------------------------------------------------------------------------------------------------------------

st.markdown(
    """ 
    
    Here we walk through the various steps used to load and clean the data. Kaggle provides 3 major synthetic tables:
    - `sd254_users.csv`: Credit card holders
    - `sd254_cards.csv`: Credit cards
    - `credit_card_transactions-ibm_v2.csv`: Transactions


    ## Credit Card holders

    """
)


# ----------------------------------------------------------------------------------------------------------------
# Sidebar 
# ----------------------------------------------------------------------------------------------------------------


users_file = st.sidebar.file_uploader("Load Credit Card Users CSV")
if users_file:
    st.session_state.users_csv_loaded = True

cards_file = st.sidebar.file_uploader("Load Credit Card Details CSV")
if cards_file:
    st.session_state.cards_csv_loaded = True

transactions_file = st.sidebar.file_uploader("Load Transactions CSV")
if transactions_file:
    st.session_state.transactions_csv_loaded = True


# ----------------------------------------------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------------------------------------------

# set the tabs
tab_users, tab_cards, tab_transactions = st.tabs(["Credit Card Users", "Credit Card Details", "Credit Card Transactions"])

# ----------------------------------------------------------------------------------------------------------------

with tab_users:
    # Before
    # load the sample users csv
    st.markdown("""
    ### The raw csv contents 
                
    The uploaded credit card user file should look like:
    """
    )
    sample_users_path = Path.cwd() / 'data/sample_users.csv'

    sample_users_df = pd.read_csv(sample_users_path)

    st.dataframe(sample_users_df.head())

    # load the users csv
    users_path = Path.cwd() / 'data/sample_users.csv'
    # sample_path = 

    st.markdown("""
    On the import, there is an opportunity to choose specific columns to load. 
    The columns we have decided to keep in this dataframe are:
    - 'Birth Year', 
    - 'Zipcode', 
    - 'Per Capita Income - Zipcode',
    - 'Yearly Income - Person', 
    - 'Total Debt',
    - 'FICO Score',
    - 'Num Credit Cards'
    
    On the import, we have preprocessed some of the columns and removed the '$' from appropriate columns and converted them into integers.
    """
    )

    load_users_button = st.button("Click to transform credit card users",
                                  disabled = not st.session_state.users_csv_loaded 
                                  )


    user_converters = {'Zipcode': add_leading_zero_to_zipcode,
                   'Per Capita Income - Zipcode': remove_dollar_and_convert,
                   'Yearly Income - Person': remove_dollar_and_convert,
                   'Total Debt': remove_dollar_and_convert}


# ----------------------------------------------------------------------------------------------------------------
