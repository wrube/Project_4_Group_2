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

if 'merged_df' not in st.session_state:
    st.session_state.merged_df = pd.DataFrame()


# ****************************************************************************************************************
# ----------------------------------------------------------------------------------------------------------------
# setup page
# ----------------------------------------------------------------------------------------------------------------

st.set_page_config(page_title="Raw Data from Kaggle", 
                   page_icon="⬇️",
                   layout='wide')

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


## Instructions
1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/ealtman2019/credit-card-transactions/data?select=credit_card_transactions-ibm_v2.csv)
2. After cloning the gitHub repo, unzip the files into the project `data` folder
3. Use the [create_dataset.ipynb](create_dataset.ipynb) file to trim the transactions file (2.5GB) into 100 and 500 user files.
4. Start using this page by ensuring the file names are correct on the sidebar.
5. Click the Merge button to begin the merge process.
6. Click through the different pages in order and let them run until completion before clicking on the next page.

    """
)


# ----------------------------------------------------------------------------------------------------------------
# Sidebar 
# ---------------------------------------------------------------------------------------------------------------


users_file = st.sidebar.text_input("Load Credit Card Users CSV",
                                   value="data/sd254_users.csv",
                                      )
if users_file:
    st.session_state.users_csv_loaded = True

cards_file = st.sidebar.text_input("Load Credit Card Details CSV",
                                   value="data/sd254_cards.csv",
                                      )
if cards_file:
    st.session_state.cards_csv_loaded = True

transactions_file = st.sidebar.text_input("Load Transactions CSV",
                                          value="data/transactions_users_100.csv",
                                      )
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

    # load_users_button = st.button("Click to transform credit card users",
    #                               disabled = not st.session_state.users_csv_loaded 
    #                               )
    # if load_users_button:

    users_path = Path.cwd() / users_file

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
    
    # load data
    users_df = load_csv_data(users_path,
                                users_columns_import,
                                user_converters,
                                users_dtypes)
        
    # add Users column
    users_df['User'] = users_df.index
        
    st.session_state.users_df = users_df

    st.markdown("### Loaded and Cleansed Users Dataframe")    
    st.dataframe(st.session_state.users_df)



# ----------------------------------------------------------------------------------------------------------------
with tab_cards:
    # Before
    # load the sample users csv
    st.markdown("""
    ### The raw csv contents 
                
    The uploaded credit card details file should look like:
    """
    )
    sample_cards_path = Path.cwd() / 'data/sample_cards.csv'

    sample_cards_df = pd.read_csv(sample_cards_path)

    st.dataframe(sample_cards_df.head())


    st.markdown("""
    On the import, there is an opportunity to choose specific columns to load. 
    The columns we have decided to keep in this dataframe are:
    - 'User',	
    - 'CARD INDEX',
    - 'Has Chip',
    - 'Cards Issued',
    - 'Year PIN last Changed',
    - 'Card on Dark Web'
    
    On the import, we have preprocessed some of the columns that contain 'Yes' or 'No' categories into the binary 1 or 0.
    """
    )

    # load_cards_button = st.button("Click to transform credit card details",
    #                               disabled = not st.session_state.cards_csv_loaded 
    #                               )
    
    # if load_cards_button:
    cards_path = Path.cwd() / cards_file

    cards_columns_import = ['User',	
                            'CARD INDEX',
                            'Has Chip',
                            'Cards Issued',
                            'Year PIN last Changed',
                            'Card on Dark Web'
                            ]

    cards_dtypes = {'CARD INDEX': np.uint8,
                    'Cards Issued': np.uint8,
                    'Year PIN last Changed': np.uint16
                    }

    cards_conversions = {'Card on Dark Web': convert_yes_no_to_binary,
                        'Has Chip': convert_yes_no_to_binary}
    
    # load data
    cards_df = load_csv_data(cards_path,
                                cards_columns_import,
                                cards_conversions,
                                cards_dtypes)
    
    st.session_state.cards_df = cards_df

        
    st.markdown("### Loaded and Cleansed Credit Cards Dataframe")    
    st.dataframe(st.session_state.cards_df)



# ----------------------------------------------------------------------------------------------------------------

with tab_transactions:
    # Before
    # load the sample users csv
    st.markdown("""
    ### The raw csv contents 
                
    The uploaded transaction details file should look like:
    """
    )
    sample_transactions_path = Path.cwd() / 'data/sample_transactions.csv'

    sample_transactions_df = pd.read_csv(sample_transactions_path)

    st.dataframe(sample_transactions_df.head())


    st.markdown("""
    On the import, there is an opportunity to choose specific columns to load. 
    The columns we have decided to keep in this dataframe are:
    - 'User',
    - 'Card',
    - 'Year',
    - 'Month',
    - 'Day',
    - 'Time',
    - 'Amount',
    - 'Use Chip',
    - 'Merchant City',
    - 'Merchant State',
    - 'Zip',
    - 'MCC',
    - 'Errors?',
    - 'Is Fraud?':  Target variable
    
    On the import, we have preprocessed some of the columns and removed the '$' from the 'Amount' column and converted it into a float. Additionally
    """
    )

    # load_transactions_button = st.button("Click to transform transaction details",
    #                               disabled = not st.session_state.transactions_csv_loaded
    #                               )
    
    # if load_transactions_button:
    transaction_path = Path.cwd() / transactions_file

    transactions_columns_import = ['User',
                                'Card',
                                'Year',
                                'Month',
                                'Day',
                                'Time',
                                'Amount',
                                'Use Chip',
                                'Merchant City',
                                'Merchant State',
                                'Zip',
                                'MCC',
                                'Errors?',
                                'Is Fraud?'
                                ]

    transaction_converters = {'Zip': add_leading_zero_to_zipcode,
                            'Amount': remove_dollar_and_convert_float,
                            # 'Is Fraud?': convert_yes_no_to_binary
                            }
    
    transaction_dtypes = {'MCC': 'category',
                          'Is Fraud?': 'category'
                          }
    
    # load data
    transactions_df = load_csv_data(transaction_path,
                                transactions_columns_import,
                                transaction_converters,
                                transaction_dtypes
                                )
    
    st.session_state.transactions_df = transactions_df

        
    st.markdown("### Loaded and Cleansed Transactions Dataframe")    
    st.dataframe(st.session_state.transactions_df)



# ----------------------------------------------------------------------------------------------------------------
# Merging into one
# ----------------------------------------------------------------------------------------------------------------

st.markdown("""
## Merging

Once all datasets are loaded and transformed, we can start merging the dataset.

This is done in 2 steps as shown below:
"""
)

merge_button = st.sidebar.button("Merge the dataframes",
                         disabled = (len(st.session_state.users_df) == 0) |
                                    (len(st.session_state.cards_df) == 0) |
                                     (len(st.session_state.transactions_df) == 0)
                         )

if merge_button:
    with st.echo():
        merge_step_1 = st.session_state.transactions_df.merge(
            st.session_state.users_df, how='inner', on='User')

        merged_df = merge_step_1.merge(st.session_state.cards_df, 
                                       how='inner', 
                                       left_on=['User', 'Card'], 
                                       right_on=['User', 'CARD INDEX'])
        st.sidebar.write('Merge Complete')
    
    # save to session state
    st.session_state.merged_df = merged_df


if len(st.session_state.merged_df) > 0:
    merged_df = st.session_state.merged_df
    st.dataframe(merged_df)


else:
    st.sidebar.markdown("#### FILES NOT YET MERGED - PLEASE MERGE")



    