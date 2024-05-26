![Logo](images/fraud.jpeg)
# *Predicting Credit Card Fraud Using Machine Learning Algorithms* 🔒🏦

## Project Overview:
Credit card fraud is a significant issue in the finnancial industry, leading to substantial financial losses and
affecting both consumers and financial institutions. The rise of digital transactions has increased the opportunities
for fraudulent activities, making it essential to develop robust and efficient methods for detecting and preventing 
fraud. This project aims to harness the power of machine learning to build an intelligent system capable of identifying 
fraudulent transactions with high accuracy.  

## Downloading Data:
To download the dataset from Kaggle using the provided link, follow these steps:

1.	Click on the link to access the Kaggle dataset page: [Credit Card Transactions Dataset](https://www.kaggle.com/datasets/ealtman2019/credit-card-transactions/data?select=credit_card_transactions-ibm_v2.csv).

2.	If you don't already have an account on Kaggle, you'll need to sign up for one. You can sign up using your Google account or create a new Kaggle account.

3.	Once logged in, you'll be directed to the dataset page. Here, you'll see a list of files available for download, including credit_card_transactions-ibm_v2.csv.

4.	Find the file you want to download (credit_card_transactions-ibm_v2.csv) and click on it. This action will initiate the download process.
5.	Depending on your browser settings, the file may download automatically to your default download location, or you may be prompted to choose a location to save the file.

6.	Once the download is complete, you can access the dataset file (credit_card_transactions-ibm_v2.csv) from the location where it was saved on your computer.

7.	You can now use this dataset for your analysis, machine learning projects, or any other purpose as needed.

![Logo](images/streamlit.png)
## Streamlit Interactive Dashboard :
1. Clone the Repository: First, clone the repository to your local machine using Git.
2. Install Dependencies: Ensure you have the necessary dependencies installed.
3. Run Streamlit App: Once the dependencies are installed, you can run your Streamlit app. Use the following command: `streamlit run <app_filename>.py`
4. Access the App: After running the above command, Streamlit will start a local development server and provide a URL where your app is hosted. Usually, it's something like http://localhost:8501. Open this URL in your web browser to access your Streamlit app.
5. Interact with the App: Once the app is running, you can interact with it in your web browser. Any changes you make to the app code will automatically update the app in real-time.
6. Stop the Server: To stop the Streamlit server, you can press Ctrl + C in the terminal where the server is running. This will stop the server and return you to the command prompt.




## Data Preparation and Tasks:
            
### About the database and further clean up  🧹
For this fraud prevention and machine learning project, we began by curating synthetic data from Kaggle, specifically
designed to simulate fraudulent transactions. This raw data was then subjected to extensive cleaning and preprocessing steps,
including the removal of irrelevant columns, handling missing values, and standardizing data formats. 
Through meticulous data engineering processes,we ensured that the dataset was streamlined and optimized for machine learning analysis.
            
More detailed information about this step is available in the `Data Load and Merge Page`- Streamlit.
            
### Further Data Engineering 🖥️
Following the cleaning phase, we merged relevant datasets to create a comprehensive dataset that encapsulates various features
and attributes related to fraudulent transactions. This merging process involved aligning data based on common identifiers
and ensuring data consistency across different sources.         
More detailed information about this step is available in the `Data Engineering Page` - Streamlit.

![GIF](images/stat.gif)

### Machine Learning and Insight 📊
Once the data was cleaned and merged, we proceeded to load it into Jupyter Notebook,
where we conducted exploratory data analysis to gain insights into the distribution and patterns within the dataset. 
This analysis helped us identify key features and trends that could be indicative of fraudulent activities. Here, we delved into exploratory
data analysis (EDA) to unearth insights into the distribution and patterns within the dataset. Utilizing various statistical techniques such as
correlation analysis, chi-squared statistics, and visualization techniques, we scrutinized the data to identify key features and trends that could
serve as indicators of fraudulent activities. To visually represent our findings and facilitate comprehension, we employed a range of visualization techniques, including bar graphs, histograms, and heatmaps.

More detailed information about this step is available in the `Data Exploration Page`- Streamlit.
                       
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
            
More detailed information about this step is available in the `Modelling Page`- Streamlit.

Overall, the project encompassed data cleaning, merging, exploratory analysis, and machine learning model development,
all aimed at building an effective fraud detection system.  

![GIF](images/scam.gif)

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

## Possible Further Improvements to the Model
While our project has successfully developed a robust fraud detection system, there are several areas where we could further enhance its performance and functionality. 
Here are some possible improvements:

* **Advanced Feature Engineering:**
            - Implement techniques like feature interaction to capture complex relationships between variables.
            - Explore temporal features to caputre time-related patterns in fraudulent transactions.

* **Hyperparameter Optimization:**
            - Utilize advanced hyperparameter tuning techniques such as Bayesian Optimization, Random Search or Genetic Algorithms to polish the models effectively.

* **Handling Imbalanced Data:**
            - Apart from using SMOTE (Synthetic Minority Over-sampling Technique) we could have tried ADASYN (Adaptive Synthetic Sampling).

* **Automated Machine Learning (AutoML):**
            - Incorporate AutoML frameworks to automate the process of model selection, hyperparameter tuning, and feature engineering, potentially uncovering better-performing models.
        
* **Robust Validation Techniques:**
            - Implement more robust cross-validation techniques, such as stratified k-fold cross-validation, to ensure our model generalizes well to unseen data.

* **Model Interpretability:** 
            - Enhancing model interpretability through techniques like SHAP (SHapley Additive exPlanations) could have helped in understanding the contribution of each feature to the model's predictions, 
            providing more insights into the factors driving fraudulent transactions.

            
Throughout our project, we primarily focused on the recall parameter for fraudulent transactions. This focus ensured that our model was not only trained to predict non-fraudulent transactions accurately
 but also excelled at identifying fraudulent ones. High recall is crucial in fraud detection, as it minimizes the chances of missing fraudulent activities, thereby improving the robustness and reliability
 of the fraud detection system. By prioritizing recall, we aimed to reduce the false negative rate, ensuring that as many fraudulent transactions as possible were correctly identified.

## Conclusion:
Our project developed a robust fraud detection system for credit card transactions using machine learning techniques. We meticulously prepared the data, conducted exploratory data analysis
to identify key patterns, and developed and evaluated multiple machine learning models. Through iterative refinement, we achieved highly accurate fraud detection capabilities, enhancing security
and trust in financial transactions.

## Ethical Considerations
The selected dataset is synthetic, ensuring the protection of individual's privacy and adhering to ethical principles. By opting for synthetic data, privacy concerns associated with the use of real world data are effectively mitigated. This choice alligns with ethical standards, prioritizing the safeguard of personal information while enabling meaningful analysis and insights to be derived.


#### Resources & Links:
[Kaggle - Transactions Dataset](https://www.kaggle.com/datasets/ealtman2019/credit-card-transactions/data?select=credit_card_transactions-ibm_v2.csv)
[Streamlit Logo](https://streamlit.io/brand)

[Cover Photo Source](https://stock.adobe.com/search/images?k=fraud+prevention&asset_id=717519915)

[Statistic GIF Source](https://dribbble.com/shots/1388718-Stats-2-GIF)

[Is this a Scam? GIF Source](https://tenor.com/en-GB/view/scam-suspicious-dubious-unconvinced-gif-17495225760777114376)

Copyright
B. Wruck, A. Czynszak, K.J. Yong, B.Osborne © 2024. All Rights Reserved.