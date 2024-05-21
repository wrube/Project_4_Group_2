# import modules
import sys
import streamlit as st
import time
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

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
st.sidebar.markdown("#### Choose your column")

if cat_column == 'Categorical':
    columns = ['Use Chip',
                'Merchant State',
                'Errors?',
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

dropdown_name = f"{cat_column} Columns"
column = st.sidebar.selectbox(dropdown_name, columns)




if len(st.session_state.final_cleaned_df) > 0:
    final_cleaned_df = st.session_state.final_cleaned_df

    st.markdown(f"### Exploring the '{column}' Feature")


    # st.write(final_cleaned_df[''])
    
    # print(column_count_df)
    # st.write(column_count_df)


# ----------------------------------------------------------------------------------------------------------------
# Categorical Display
# ----------------------------------------------------------------------------------------------------------------

    if cat_column == 'Categorical':
        # st.write(final_cleaned_df[[column, 'Is Fraud?']].groupby(columns).value_counts())
        
        # Group by 'column' and Is Fraud, then count occurrences
        pivot_df = (final_cleaned_df.pivot_table(index=column, 
                                              columns='Is Fraud?', 
                                              aggfunc='size')
                                .rename(columns={'0': 'Not Fraud',
                                                '1': 'Fraud'})
                                .sort_values('Fraud', ascending=False)
                )
        
        x_pos = np.arange(len(pivot_df.iloc[0:10, 1]))
        x_categories = pivot_df.iloc[0:10, 1].index.values
        
        # print the table
        st.dataframe(pivot_df)

        # plot bar chart
        fig, ax = plt.subplots()
        ax.bar(x_pos, pivot_df.iloc[0:10, 1], mouseover=True)
        ax.set_xlabel(f"{column}")
        if len(x_pos) > 5:
            ax.set_xticks(x_pos, x_categories, rotation = 60)
        else:
            ax.set_xticks(x_pos, x_categories)
        ax.set_ylabel('Count')
        ax.set_title(f"Top Fraudulent Transactions by {column}")

        st.pyplot(fig)
        

    
    

    else:
        st.write(final_cleaned_df[column].describe())


