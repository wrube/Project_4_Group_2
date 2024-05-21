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





# ****************************************************************************************************************
# ----------------------------------------------------------------------------------------------------------------
# setup page
# ----------------------------------------------------------------------------------------------------------------

st.set_page_config(page_title="Feature Engineering", 
                   page_icon="ℹ️")

# ----------------------------------------------------------------------------------------------------------------
# page content
# ----------------------------------------------------------------------------------------------------------------

st.markdown(""" 
# Feature Engineering
            
Additional features can be extracted from the dataset which we deemed useful to the analysis:
            
- "International" feature for international transactions
- "Online" feature for online transactions 
- "Age_at_transaction"
- "income_to_debt" feature to provide some information whether the user is over-leveraged
- "datetime" feature which temporarily exists to provide information on the day of the week
            """)

if 'merged_df' in st.session_state:
    merged_df = st.session_state.merged_df
    with st.echo():
        # 

        # add column to define whether international
        merged_df['International'] = (merged_df['Merchant State'].str.len() > 2).astype(np.int8)

        # add column for online transaction
        merged_df['Online'] = (merged_df['Merchant City'] == 'ONLINE').astype(np.int8)

        # add column for age at transaction
        merged_df['Age_at_transaction'] = (merged_df['Year'] - merged_df['Birth Year']).astype(np.int16)

        # create income-to-debt
        merged_df['income_to_debt'] = merged_df['Yearly Income - Person'] / (merged_df['Total Debt'] + 0.001)

        # create a date-time column
        merged_df['datetime'] = pd.to_datetime(merged_df['Year'].astype(str) + '-' + 
                                        merged_df['Month'].astype(str) + '-' + 
                                        merged_df['Day'].astype(str) + ' ' + 
                                        merged_df['Time'])

        # create day of week column
        merged_df['day_of_week'] = merged_df['datetime'].dt.dayofweek.astype(np.int8)

        # Convert datetime to Unix timestamp
        merged_df['timestamp'] = merged_df['datetime'].astype(int) / 10**9  # Convert to seconds

        # Put times into bins of time-of-day

        # Define the bins and their corresponding labels
        time_bins = [0, 6, 12, 18, 22, 24]
        time_labels = ['Night', 'Morning', 'Afternoon', 'Evening', 'Night']

        # Categorize the hours into bins
        merged_df['time_of_day'] = pd.cut(merged_df['datetime'].dt.hour, bins=time_bins, labels=time_labels, right=False, include_lowest=True, ordered=False)

st.markdown(""" 

            """)