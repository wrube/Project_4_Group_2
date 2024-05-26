
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
# session state variables
# ----------------------------------------------------------------------------------------------------------------

if 'users_df' not in st.session_state:
    st.session_state.users_df = pd.DataFrame()

if 'cards_df' not in st.session_state:
    st.session_state.cards_df = pd.DataFrame()

if 'transactions_df' not in st.session_state:
    st.session_state.transactions_df = pd.DataFrame()

# ----------------------------------------------------------------------------------------------------------------
# setup page
# ----------------------------------------------------------------------------------------------------------------

st.set_page_config(
    page_title="Detecting Credit Card Fraud",
    page_icon="💳",
)

# ----------------------------------------------------------------------------------------------------------------
st.image("images/fraud.jpeg", use_column_width=True)
st.write("# *Predicting Credit Card Fraud Using Machine Learning Algorithms* 🔒🏦 ")

# st.sidebar.success("Select in Order")


st.markdown("""
## Project Overview:
Credit card fraud is a significant issue in the finnancial industry, leading to substantial financial losses and
affecting both consumers and financial institutions. The rise of digital transactions has increased the opportunities
for fraudulent activities, making it essential to develop robust and efficient methods for detecting and preventing 
fraud. This project aims to harness the power of machine learning to build an intelligent system capable of identifying 
fraudulent transactions with high accuracy.  


## Data Preparation and Tasks:
            
### About the database and further clean up  🧹
For this fraud prevention and machine learning project, we began by curating synthetic data from Kaggle, specifically
designed to simulate fraudulent transactions. This raw data was then subjected to extensive cleaning and preprocessing steps,
including the removal of irrelevant columns, handling missing values, and standardizing data formats. 
Through meticulous data engineering processes,we ensured that the dataset was streamlined and optimized for machine learning analysis.
            
More detailed information about this step is available in the `Data Load and Merge Page`.
            
### Further Data Engineering 🖥️
Following the cleaning phase, we merged relevant datasets to create a comprehensive dataset that encapsulates various features
and attributes related to fraudulent transactions. This merging process involved aligning data based on common identifiers
and ensuring data consistency across different sources.         
More detailed information about this step is available in the `Data Engineering Page`.
"""
)  
st.image("images/stat.gif", width = 400)
st.markdown("""
### Machine Learning and Insight 📊
Once the data was cleaned and merged, we proceeded to load it into Jupyter Notebook,
where we conducted exploratory data analysis to gain insights into the distribution and patterns within the dataset. 
This analysis helped us identify key features and trends that could be indicative of fraudulent activities. Here, we delved into exploratory
data analysis (EDA) to unearth insights into the distribution and patterns within the dataset. Utilizing various statistical techniques such as
correlation analysis, chi-squared statistics, and visualization techniques, we scrutinized the data to identify key features and trends that could
serve as indicators of fraudulent activities. To visually represent our findings and facilitate comprehension, we employed a range of visualization techniques,
 including bar graphs, histograms, and heatmaps.

More detailed information about this step is available in the `Data Exploration Page`.
                       
### Further Machine Learning and Model Prediction 🧠
With the prepared dataset, we then embarked on the development of machine learning models for fraud detection. 
This involved tasks such as feature engineering, model selection, hyperparameter tuning, and model evaluation.
Throughout this process, we iteratively refined our models to achieve optimal performance in accurately identifying
and predicting fraudulent transactions. We developed and tested several models, including **Linear Regression, XGBoost, Random Forest, and Neural Network**. 
Each model was meticulously crafted and fine-tuned to extract actionable insights from the data and accurately predict fraudulent transactions.
            
Throughout this iterative process, we diligently assessed the efficacy of each model, meticulously scrutinizing key performance metrics including accuracy, precision,
recall, and F1-score. This rigorous evaluation empowered us to pinpoint the most suitable model tailored to our unique use case and fine-tune it for enhanced predictive prowess.

Employing a holistic approach to both model development and evaluation, we successfully deployed resilient fraud detection systems. These systems boast a high level of accuracy
and reliability, effectively identifying and thwarting fraudulent activities with precision and confidence.
            
More detailed information about this step is available in the `Modelling Page`.

Overall, the project encompassed data cleaning, merging, exploratory analysis, and machine learning model development,
all aimed at building an effective fraud detection system.   
"""
)
st.image("images/scam.gif", width=400)

st.markdown(""" 
## Fraud Detection and Model Evaluation Interactive Dashboard:
This Python file serves as a tool for fraud detection and model evaluation in a machine learning environment. It was designed for the sole purpose of:

* Model Deployment: After training a fraud detection model, organizations often need a convenient way to deploy the model for real-time prediction or batch processing.
 Our script provides a user-friendly interface for making predictions on new data.

* Model Evaluation: It's essential to continuously evaluate the performance of a deployed model to ensure its effectiveness and reliability. Our script enables users to
 assess the model's accuracy and other metrics using both summary statistics and visualizations like confusion matrices and feature importance plots.

* Feature Analysis: Understanding which features contribute most to the model's predictions is crucial for improving model interpretability and identifying potential areas for
 feature engineering. Our script offers functionality to display feature importance, helping users gain insights into the underlying data patterns.

* Interactive Exploration: By providing a menu-driven interface, the script allows users to interactively explore model predictions, evaluate model performance, and visualize feature
 importance without the need for complex coding or data manipulation.        

This script empowers users to interactively explore and analyze fraud detection models, facilitating informed decision-making in fraud prevention efforts.

## Conclusion:
Our project developed a robust fraud detection system for credit card transactions using machine learning techniques. We meticulously prepared the data, conducted exploratory data analysis
to identify key patterns, and developed and evaluated multiple machine learning models. Through iterative refinement, we achieved highly accurate fraud detection capabilities, enhancing security
and trust in financial transactions.

"""
)