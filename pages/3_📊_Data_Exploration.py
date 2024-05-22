# import modules
import sys
import streamlit as st
import time
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import chi2_contingency

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
                   page_icon="📊",
                   layout='wide')


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
        relative_freq_df = pivot_df.copy()
        relative_freq_df['Not Fraud'] = pivot_df['Not Fraud'] / pivot_df.sum(axis=1) * 100
        relative_freq_df['Fraud'] = pivot_df['Fraud'] / pivot_df.sum(axis=1) * 100
        relative_freq_df = relative_freq_df.sort_values('Fraud', ascending=False)

        max_categories = 7


        col1, col2 = st.columns(2, gap='medium')

        with col1:
            st.markdown("### Raw Counts")
            # print the table
            st.dataframe(pivot_df)

        with col2:
            st.markdown("### Percent Relative Frequency")
            # print the table
            st.dataframe(relative_freq_df)
            
        # plot bar chart
        x_categories = pivot_df.iloc[0:max_categories, 1].index.values
        x_pos = np.arange(len(pivot_df.iloc[0:max_categories, 1]))

        fig, (ax1, ax2) = plt.subplots(1, 2,
                                        width_ratios = [0.5, 0.5])
        ax1.bar(x_pos, pivot_df.iloc[0:max_categories, 1], mouseover=True)
        ax1.set_xlabel(f"{column}")
        if len(x_pos) > 2:
            ax1.set_xticks(x_pos, x_categories, rotation = 80)
        else:
            ax1.set_xticks(x_pos, x_categories)
        
        ax1.set_xticklabels(x_categories, fontsize=8)

        ax1.set_ylabel('Count')
        ax1.set_title(f"Top Fraudulent Transactions \n by {column}",
                      fontsize=10)

        # plot bar chart relative frequencies
        x_rel_categories = relative_freq_df.iloc[0:max_categories, 1].index.values

        ax2.bar(x_pos, relative_freq_df.iloc[0:max_categories, 1], mouseover=True)
        ax2.set_xlabel(f"{column}")

        if len(x_pos) > 2:
            ax2.set_xticks(x_pos, x_rel_categories, rotation = 80)
        else:
            ax2.set_xticks(x_pos, x_rel_categories)

        ax2.set_xticklabels(x_rel_categories, fontsize=8)
        ax2.set_ylabel('Percent (%)', fontsize=8)
        ax2.set_title(f"Top Relative Frequencies of Fraudulent \n Transactions by {column}",
                      fontsize=10)

        fig.set_figheight(5)
        fig.tight_layout()

        st.pyplot(fig)
        
        st.markdown(f"### Statistical Significance of '{column}' Feature
                    
        Using the chi-squared test, we can test for the significance of this category. 
        Here we use the scipy package for the calculation  
                    ")
        # statistical chi-squared test
        # Performing chi-squared test of independence

        with st.echo():
            chi2, p_value, dof, expected = chi2_contingency(pivot_df)

        st.write(f"Chi-squared statistic: {chi2}")
        st.write(f"P-value:  {p_value}")
        st.write(expected)

    
# ----------------------------------------------------------------------------------------------------------------
# Continuous Display
# ----------------------------------------------------------------------------------------------------------------


    else:
        st.write(final_cleaned_df[column].describe())


