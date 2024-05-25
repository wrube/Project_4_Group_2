# import modules
import sys
import streamlit as st
import time
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import chi2_contingency, ttest_ind

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
                              options=["Numerical", "Categorical"],
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
                'time_of_day',
                'Is Fraud?']
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

    st.markdown(f"## Exploring the '{column}' Feature")

# ----------------------------------------------------------------------------------------------------------------
# Categorical Display
# ----------------------------------------------------------------------------------------------------------------

    if cat_column == 'Categorical':
        st.write(final_cleaned_df) #[[column, 'Is Fraud?']].groupby(columns).value_counts())
        st.write(final_cleaned_df.index)
        st.write(column)
        initial_pivot_df =  final_cleaned_df.pivot_table(index=column, columns='Is Fraud?', aggfunc='size')
        st.write(initial_pivot_df)
        #st.write([(type(c), c) for c in initial_pivot_df.columns])
        renamed_pivot_df = initial_pivot_df.rename({0: 'Not Fraud', 1: 'Fraud'}, axis='columns')
        #st.write([(type(c), c) for c in renamed_pivot_df.columns])
        # Group by 'column' and Is Fraud, then count occurrences
        pivot_df = renamed_pivot_df.sort_values('Fraud', ascending=False)
        #st.write(pivot_df)
        relative_freq_df = pivot_df.copy()
        relative_freq_df['Not Fraud'] = pivot_df['Not Fraud'] / pivot_df.sum(axis=1) * 100
        relative_freq_df['Fraud'] = pivot_df['Fraud'] / pivot_df.sum(axis=1) * 100
        relative_freq_df = relative_freq_df.sort_values('Fraud', ascending=False)

        max_categories = 7

        st.markdown(f"""
### Statistical Significance of '{column}' Feature
                    
Using the chi-squared test, we can test for the significance of this category. 
Here we use the scipy package for the calculation  
                    """)
        
        # statistical chi-squared test
        # Performing chi-squared test of independence

        with st.echo():
            chi2, p_value, dof, expected = chi2_contingency(pivot_df)

        st.write(f"Chi-squared statistic: {chi2:.2f}")
        st.write(f"P-value:  {p_value:.2f}")

        expected_df = pd.DataFrame(expected)
        

        col1, col2= st.columns(2, gap='medium')

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

        
# ----------------------------------------------------------------------------------------------------------------
# Numerical Display
# ----------------------------------------------------------------------------------------------------------------


    else:
        # # Separate data for 'Is Fraud' = 0 and 'Is Fraud' = 1
        # not_fraud_df = final_cleaned_df.loc[final_cleaned_df['Is Fraud?'] == 0]
        not_fraud_df = final_cleaned_df.loc[final_cleaned_df['Is Fraud?'] == '0', column]

        fraud_df = final_cleaned_df.loc[final_cleaned_df['Is Fraud?'] == '1', column]

        # category_df = final_cleaned_df[[column, 'Is Fraud?']].groupby('Is Fraud?')
        st.markdown("## Descriptive Statistics Summaries")

        col1, col2, col3 = st.columns([0.25, 0.25, 0.5])

        with col1:
            st.markdown("### Not Fraud")
            # print the table
            st.write(not_fraud_df.describe())

        with col2:
            st.markdown("### Is Fraud")
            # print the table
            st.dataframe(fraud_df.describe())

        with col3:
            st.markdown("""
### Statistical Significance

Using the two sample Welech's t-test we can test for significance betweens the 
means of 'Fraud' and 'Not Fraud' of this category. 

Here we use the scipy package for the calculation                        
            """    
            )
            with st.echo():
                t_test = ttest_ind(not_fraud_df, 
                                   fraud_df, 
                                   equal_var=False)

            st.write(f"t-statistic: {t_test.statistic:.2f}")
            st.write(f"P-value:  {t_test.pvalue:.2f}")

        # Plots
        fig2, (ax1, ax2) = plt.subplots(1, 2, tight_layout=True,   )


        # set bin size depending on numeric type
        if not_fraud_df.dtype in ['int8', 'uint8']:
            n_bins=5
        elif not_fraud_df.dtype in ['int16', 'uint16']:
            n_bins=10
        else:
            n_bins = 50

        ax1.hist(not_fraud_df, bins=n_bins, label='Not Fraud', alpha=0.4, density=True)
        ax1.hist(fraud_df, bins=n_bins, label='Fraud', alpha=0.4, density=True)
        ax1.legend(shadow=True)
        ax1.set_xlabel(f"{column}")
        ax1.set_ylabel("Density")
        ax1.set_title(f"Histogram Analysis Between Fraud and \n Not Fraud", fontsize=10)

        ax2.boxplot([not_fraud_df.values, fraud_df.values],
                   labels=['Not Fraud', 'Fraud'])
        ax2.set_ylabel(f"{column}")
        ax2.set_title(f"Boxplot Comparison of Fraud and \n Not Fraud in {column}", fontsize=10)

        fig2.set_figheight(3)
        st.pyplot(fig2)

    fraud_by_day_time = final_cleaned_df.groupby(['day_of_week', 'time_of_day'])['Is Fraud?'].mean().reset_index()
    st.dataframe(fraud_by_day_time)

    # Pivotting the DataFrame to create a matrix suitable for heatmap
    heatmap_data = fraud_by_day_time.pivot(index='day_of_week', columns='time_of_day', values='Is Fraud?')

    # Plotting the heatmap
    fig3, ax = plt.subplots()
    sns.heatmap(heatmap_data, cmap='coolwarm', annot=True, fmt=".2%", cbar_kws={'label': 'Mean Fraud Occurrence'})
    ax.set_title('Fraud Occurrence by Day of the Week and Time of the Day')
    ax.set_xlabel('Time of Day')
    ax.set_ylabel('Day of the Week')

    # Display the heatmap
    st.pyplot(fig3)
   
    