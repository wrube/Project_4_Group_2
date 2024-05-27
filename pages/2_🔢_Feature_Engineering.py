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

# if 'final_cleaned_df' not in st.session_state:
#     st.session_state.final_cleaned_df = pd.DataFrame()

# dataframe cleaned except has a User column
if 'nearly_final_df' not in st.session_state:
    st.session_state.nearly_final_df = pd.DataFrame()

if 'merged_df' not in st.session_state:
    st.session_state.merged_df = pd.DataFrame()

if 'training_features_df' not in st.session_state:
    st.session_state.training_features_df = pd.DataFrame()

if 'training_target_df' not in st.session_state:
    st.session_state.training_target_df = pd.DataFrame()

if 'test_features_df' not in st.session_state:
    st.session_state.test_features_df = pd.DataFrame()

if 'test_target_df' not in st.session_state:
    st.session_state.test_target_df = pd.DataFrame()

if 'n_users_for_train_test' not in st.session_state:
    st.session_state.n_users_for_train_test = 5


# Callback function to update session state
def update_slider_value():
    st.session_state.n_users_for_train_test = st.session_state.n_users_slider



# ****************************************************************************************************************
# ----------------------------------------------------------------------------------------------------------------
# setup page
# ----------------------------------------------------------------------------------------------------------------
st.set_page_config(page_title="Feature Engineering", 
                   page_icon="🔢",
                   layout='wide')

if len(st.session_state.merged_df) > 0:

    # alias for session_state.merged_df
    merged_df = st.session_state.merged_df




# ----------------------------------------------------------------------------------------------------------------
# Page Options On Sidebar
# ----------------------------------------------------------------------------------------------------------------

    st.sidebar.markdown("## Feature Engineering Output Options")
    n_users = merged_df['User'].nunique()

    st.sidebar.markdown(f"There are **{n_users}** unique users in this dataset")

# ----------------------------------------------------------------------------------------------------------------
# Use whole dataframe for Training and Test?
# ----------------------------------------------------------------------------------------------------------------
    st.sidebar.markdown("#### Dataset Options For Training and Test Sets")


    
    n_users_for_train_test = st.sidebar.select_slider("Select the number of users for training and test Set?", 
                             options=np.arange(1, n_users, 1),
                             value=st.session_state.n_users_for_train_test,
                             key='n_users_slider',
                             on_change = update_slider_value,
                            )

    
    testing_fraction = st.sidebar.select_slider("What fraction of the dataset is for Testing",
                                                options=np.arange(0.05, 0.61, 0.01),
                                                value=0.2,
                                                format_func=lambda x: f"{x:.2f}")


# ----------------------------------------------------------------------------------------------------------------
# Saving Dataframe option
# ----------------------------------------------------------------------------------------------------------------


    save_cleaned_df = st.sidebar.toggle("Save Cleaned Dataframe?", value=False)

    if save_cleaned_df:

        st.sidebar.markdown("### Option to Save Cleaned Dataframe")



        first_user, last_user = st.sidebar.slider("Range of Users to Save", 
                                                min_value=0,
                                                max_value=n_users - 1,
                                                value=(0, n_users - 1)
                                                )
        
        if len(st.session_state.nearly_final_df) > 0:
            nearly_final_df = st.session_state.nearly_final_df
            
            trimmed_df = nearly_final_df.loc[(nearly_final_df['User'] >= first_user)  & 
                                             (nearly_final_df['User'] < last_user)].drop('User', axis=1)
            
            trimmed_df.loc[:, 'Is Fraud?'] = trimmed_df['Is Fraud?'].cat.codes
            
            save_name = st.sidebar.text_input("Dataframe save name", 
                                              value=f"cleansed_dataset_users_{first_user}-{last_user}.csv")
            
            st.dataframe(trimmed_df)

            
            save_button = st.sidebar.button("Save to csv (in 'data' directory)")
            if save_button:
                trimmed_df.to_csv(Path.cwd() / "data" / save_name, index=False)

# ----------------------------------------------------------------------------------------------------------------
# Splitting off Training and Test Data
# ----------------------------------------------------------------------------------------------------------------


   
# ----------------------------------------------------------------------------------------------------------------
# page content
# ----------------------------------------------------------------------------------------------------------------


    st.markdown(""" 
# 🔢 Feature Engineering
            
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
        merged_df['timestamp'] = merged_df['datetime'].astype(np.int64) / 10**9  # Convert to seconds

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

        # ----------------------------------------------------------------------------
        # Additional filtering
        # ----------------------------------------------------------------------------
        # Filter out negative amounts
        merged_df = merged_df.loc[merged_df['Amount'] > 0.0]

        # replace NaN Errors with No Error
        merged_df.loc[:, 'Errors?'] = merged_df['Errors?'].fillna('No Error')

        # replace NaN in Merchant State with Online
        merged_df.loc[:, 'Merchant State'] = merged_df['Merchant State'].fillna('Online')

    


# ----------------------------------------------------------------------------------------------------------------
# Dropping columns and final clean
# ----------------------------------------------------------------------------------------------------------------


    st.markdown("""
### Dropping Columns and Final Clean

Here we drop the following columns:

| Feature                   | Reason |
|:-------------------------|:--------|
|Card                        | Card index number not predictive. Was used for merge.
|Year                        | Rolled into datetime
|Month                       | Rolled into datetime
|Birth Year                  | Used for 'Age at transaction'
|Day                         |  Rolled into datetime
|Time                        |  Rolled into datetime and binned
|Merchant City               | Too much granularity 
|Zip                         | Used in distance calculation. Too much granularity to keep
|Zipcode                     |  Used in distance calculation. Too much granularity to keep
|CARD INDEX                  | Not predictive
|Year PIN last Changed       | Tried to feature engineer this but not used in this project
|MCC                         | Could potentially use, but not used in this project
|datetime                    | Discarded after extracting day of week and making an integer form of this
|Card on Dark Web            | Could be very useful, however in our datasets, all of this was 0, therefore dropped
    """
    )
    
    columns_to_drop = ['Card',
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

    st.session_state.nearly_final_df = merged_and_drop_df.reset_index()





# ----------------------------------------------------------------------------------------------------------------
# Saving Training and Test
# ----------------------------------------------------------------------------------------------------------------
    # Define the target variable
    target = 'Is Fraud?'
    nearly_final_df = st.session_state.nearly_final_df

    # Split the data into training and testing sets
    X = (nearly_final_df.loc[nearly_final_df['User']  < n_users_for_train_test]
                                        .drop(columns=['User', target], axis=1))
    
    y = nearly_final_df.loc[nearly_final_df['User'] < n_users_for_train_test][target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=testing_fraction, 
                                                        random_state=42)

    st.session_state.training_features_df = X_train
    st.session_state.training_target_df = y_train
    st.session_state.test_features_df = X_test
    st.session_state.test_target_df = y_test
    

st.markdown("")  
if len(st.session_state.training_features_df) > 0:
    st.dataframe(st.session_state.training_features_df)




    