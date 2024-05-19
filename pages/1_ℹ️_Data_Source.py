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
# Tabs
# ----------------------------------------------------------------------------------------------------------------



tab_users, tab_cards, tab_transactions = st.tabs(["Credit Card Users", "Credit Card Details", "Credit Card Transactions"])


with tab_users:
    # Before
    # load the sample users csv
    st.markdown("""
    ## Before 
    """
    )
    users_path = Path.cwd() / 'data/sample_users.csv'

    sample_users_df = pd.read_csv(users_path)

    st.dataframe(sample_users_df.head())

    # load the users csv
    users_path = Path.cwd() / 'data/sample_users.csv'
    # sample_path = 

    st.markdown("""
    As can be seen columns, there needs to be a clean up!.......
    """
    )
