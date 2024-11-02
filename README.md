# ML_credit_card_fraud_detection
Detection of Fraudulent Credit Card Transactions
This is a simple implementation of the model saved by the name "xgboost_fraud_detection.model" which is trained to predict fraudulent and legit credit card transactions. 
The second file "xgbr_model.json" is used to predict the remaining 19 input values other than the 6 input provided by the user.
The third file is app.py which contains the implementation of model in the form of an streamlit app.
Dependencies: 1. install latest version of python from "https://www.python.org/downloads/".
              2. run "pip install streamlit numpy xgboost" in the terminal to install the dependencies.
Download all the 3 files in the repository.
Open the project folder where u downloaded all the 3 files and in the terminal run "streamlit run app.py". (Open terminal in the same folder where the files are downloaded).
Now you can predict the transaction, whether fraud or legit based on the inputs you give.
