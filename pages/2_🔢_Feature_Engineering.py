# import modules
import sys
import streamlit as st
import time
import numpy as np
from pathlib import Path
import pandas as pd
import pgeocode


# import modules
sys.path.append(Path.cwd().parent)
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
st.set_page_config(page_title="Feature Engineering", 
                   page_icon="🔢",
                   layout='wide')

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

**Please note that below will remain blank until the dataset is merged from the previous page**            
""")

# ----------------------------------------------------------------------------------------------------------------
# Feature Engineering
# ----------------------------------------------------------------------------------------------------------------


if len(st.session_state.merged_df) > 0:
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
    ### Adding a Distance Feature

    Intuitively, we know that spontaneous transactions that are far away from the users home could 
    potentially be fraudulent. So we decided to use a distance attribute which could be calculated
    from between the credit card users zipcode and the merchants zipcode. 
    
    This calculation was done by using the Python package [pgeocode](https://pypi.org/project/pgeocode/).
                
    Upon using this technique, it became apparent that several zipcodes in the dataset
    does not exist. Thus manual lookups of alternate zipcodes for the merchant cities
    was done from [https://tools.usps.com/zip-code-lookup.htm](https://tools.usps.com/zip-code-lookup.htm).
    The results were embedded within the csv load function for the transaction dataset.
    
                """)
    
    with st.echo():
        
        # st.write("Create filters and lists")

        # create filters and lists of zipcodes to find distance with
        distance_candidates = merged_df['Zip'].str.len() == 5
        international_filters = merged_df['International'] == 1
        online_filters = merged_df['Online'] == 1


        merchant_zip_list = merged_df.loc[distance_candidates, 'Zip'].to_list()
        user_zip_list = merged_df.loc[distance_candidates, 'Zipcode'].to_list()

        # use pgeocode to calculate distances
        dist = pgeocode.GeoDistance('us')
        distances = dist.query_postal_code(user_zip_list, merchant_zip_list)

        avg_distance = np.mean(distances)
        max_distance = np.max(distances)


        # initiate distance attribute with average distance
        merged_df['distances'] = avg_distance

        # populate distances
        merged_df.loc[distance_candidates, 'distances'] = distances
        merged_df.loc[international_filters, 'distances'] = max_distance

# ----------------------------------------------------------------------------------------------------------------
# Dropping columns and final clean
# ----------------------------------------------------------------------------------------------------------------


    st.markdown("""
    ### Dropping Columns and Final Clean
    
    Here 
    """
    )
    
    columns_to_drop = ['Card',
                   'User',
                   'Year',
                   'Month',
                   'Birth Year',
                   'Day',
                   'Time',
                   'Merchant City',
                   'Zip',
                   'Zipcode',
                   'CARD INDEX',
                   'Year PIN last Changed',
                   'MCC',
                   'datetime',
                   'Card on Dark Web'
                   ]
    

    # drop the columns
    merged_and_drop_df = merged_df.drop(columns=columns_to_drop, axis=1)

    # Filter out negative amounts
    merged_and_drop_df = merged_and_drop_df.loc[merged_and_drop_df['Amount'] > 0.0]

    # replace NaN Errors with No Error
    merged_and_drop_df['Errors?'] = merged_and_drop_df['Errors?'].fillna('No Error')

    # replace NaN in Merchant State with Online
    merged_and_drop_df['Merchant State'] = merged_and_drop_df['Merchant State'].fillna('Online')

    st.session_state.final_cleaned_df = merged_and_drop_df

st.markdown("")  
if len(st.session_state.final_cleaned_df) > 0:
    st.dataframe(st.session_state.final_cleaned_df)


    