
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

st.set_page_config(
    page_title="Detecting Credit Card Fraud",
    page_icon="💳",
)

st.write("# Welcome to Project 4 Group 2: We're Solving Credit Card Fraud")

# st.sidebar.success("Select in Order")


st.markdown("""
We're trying to predict fraudlent credit-card transactions using a synthetic dataset from Kaggle.
"""
)

