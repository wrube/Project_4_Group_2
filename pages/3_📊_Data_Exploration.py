# import modules
import sys
import streamlit as st
import time
import numpy as np
from pathlib import Path
import pandas as pd

#import modules
# sys.path.append(Path.cwd().parent)
from proj_modules import *


# ----------------------------------------------------------------------------------------------------------------
# session state variables
# ----------------------------------------------------------------------------------------------------------------

if 'final_cleaned_df' not in st.session_state:
    st.session_state.final_cleaned_df = pd.DataFrame()


# ****************************************************************************************************************
# ----------------------------------------------------------------------------------------------------------------
# setup page
# ----------------------------------------------------------------------------------------------------------------

st.set_page_config(page_title="Data Exploration", 
                   page_icon="🔢")


# ----------------------------------------------------------------------------------------------------------------
# page content
# ----------------------------------------------------------------------------------------------------------------

st.markdown(""" # Data Exploration

            

            """)

cat_column = st.sidebar.radio("Data Type",
                              options=["Continuous", "Categorical"],
                              horizontal=True
                                )


st.markdown(""" ### Choose your column
            """)





if len(st.session_state.final_cleaned_df) > 0:
    final_cleaned_df = st.session_state.final_cleaned_df
    col
    if cat_column == 'Continuous':
        columns = ['Use Chip',
                    'Merchant State',
                    'Errors?',
                    'Is Fraud?',
                    'Has Chip',
                    'International',
                    'Online',
                    'day_of_week',
                    'time_of_day']
    else:
        columns = ['Amount',
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

