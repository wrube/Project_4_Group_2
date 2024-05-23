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


# setup page
st.set_page_config(page_title="Load and Transform Credit Card Users", page_icon="📈")


# page content
st.markdown("# Credit")
st.sidebar.header("Extract Transform and Load")


# Before
# load the users csv
st.markdown("""
## Before 
"""
)
users_path = Path.cwd() / 'data/sample_users.csv'

sample_users_df = pd.read_csv(users_path)

st.dataframe(sample_users_df.head(20))


# After
st.markdown("""
## After 
"""
)

users_columns_import = ['Birth Year', 
                        'Zipcode', 
                        'Per Capita Income - Zipcode',
                        'Yearly Income - Person', 
                        'Total Debt',
                        'FICO Score',
                        'Num Credit Cards']

user_converters = {'Zipcode': add_leading_zero_to_zipcode,
                   'Per Capita Income - Zipcode': remove_dollar_and_convert,
                   'Yearly Income - Person': remove_dollar_and_convert,
                   'Total Debt': remove_dollar_and_convert}

users_dtypes = {'Birth Year': np.uint16,
                 'FICO Score': np.uint16,
                 'Num Credit Cards': np.uint8}


users_file = st.sidebar.file_uploader("Load Credit Card Users CSV")
if users_file is not None:

# users_path = Path.cwd() / 'data/sample_users.csv'
    users_df = pd.read_csv(users_file, 
                        usecols=users_columns_import,
                        converters=user_converters,
                        dtype=users_dtypes
                        )


    users_df['User'] = users_df.index

    st.dataframe(users_df.head())