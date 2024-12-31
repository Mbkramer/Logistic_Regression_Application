# This is a practice application of logistic regression using Kaggle telocom customer data to predict customer churn
# Inspired by learnings from code tutorial and logistic regression application python file
# Dataset link: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
# Writtern by Max Kramer
# Date: 12/30/24

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Store telecom customer data as a global data frame
df = pd.read_csv('Datasets/telco_customer_churn.csv')

# Perpare telecom data for analysis and logsitic regression
def prepare_data_for_analysis():

    print(df.info())

    # Drop customer id column and rename gender and tenure columns for consistency
    # Removed nulls from TotalCharges
    df.drop(columns={"customerID"}, axis=1, inplace=True)
    df.rename(columns={"gender": "Gender", "tenure": "Tenure"}, inplace=True)

    print(df.info())

    # Perform essential transformations of binary variables
    # Look for comments that that highlight what classifications were made to convert categorical fields to binary
    # TotalCharges converted from object to float64 
    df.Gender = [1 if value =="Female" else 0 for value in df.Gender]
    df.Partner = [1 if value =="Yes" else 0 for value in df.Partner]
    df.Dependents = [1 if value =="Yes" else 0 for value in df.Dependents]
    df.PhoneService = [1 if value =="Yes" else 0 for value in df.PhoneService]
    df.MultipleLines = [1 if value =="Yes" else 0 for value in df.MultipleLines] # No and No Phone Servuce will both be 0
    df.InternetService = [0 if value =="No" else 1 for value in df.InternetService] # DSL and Fiber optic will both be 1
    df.OnlineSecurity = [1 if value =="Yes" else 0 for value in df.OnlineSecurity] # No and No internet service will both be 0
    df.OnlineBackup = [1 if value =="Yes" else 0 for value in df.OnlineBackup] # No and No internet service will both be 0
    df.DeviceProtection = [1 if value =="Yes" else 0 for value in df.DeviceProtection] # No and No internet service will both be 0
    df.TechSupport = [1 if value =="Yes" else 0 for value in df.TechSupport] # No and No internet service will both be 0
    df.StreamingTV = [1 if value =="Yes" else 0 for value in df.StreamingTV] # No and No internet service will both be 0
    df.StreamingMovies = [1 if value =="Yes" else 0 for value in df.StreamingMovies] # No and No internet service will both be 0
    df.Contract = [1 if value =="Month-to-month" else 0 for value in df.Contract] # Month-to-month will be 1 while and One year and Two year contracts will be 0
    df.PaperlessBilling = [1 if value =="Yes" else 0 for value in df.PaperlessBilling] 
    df.PaymentMethod = [1 if (value =="Bank transfer (automatic)" or value =="Credit card (automatic)") else 0 for value in df.PaymentMethod] #Payment method will be strictly defining whether paymnet is automated
    df["TotalCharges"] = df['TotalCharges'].astype('float') #Total charges converted from object to float
    df.Churn = [1 if value =="Yes" else 0 for value in df.Churn]
    
    print(df.head())
    print(df.info())

def exploratory_data_analysis():

    # Heatmap to compare all variables within the diabetes dataframe.
    plt.figure(figsize=(16, 12))
    sns.heatmap(df.corr(), annot=True, linewidths=2)
    plt.show()

def logistic_regression():

    print("\nNow that the data has been prepared, the churn column will be removed from the predictor variables to train a prediction model.")
    print("In order to fit a logistc regression to the predictors data, the predictors will be scaled to ensure that they are comparably impactful.")
    print("After the predictors are scaled, both the predictors and target data will be split into train and test the model to assess how well it predicts a customer churn. \n")
    
    y = df["Churn"] # Target variables are moved into a single column target df
    X = df.drop(["Churn"], axis=1) # Independent variables are moved into a predictors df

    # Normalize units.. lots of variance in data so that variables are comparably impactful
    scaler = StandardScaler()

    # Fit and transform
    X_scaled = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=16)
    lr = LogisticRegression()

    # Train
    lr.fit(X_train, y_train)

    # Predict on test data
    y_pred = lr.predict(X_test)

    # Evaluation variables and metrics: Accuracy, Recall, Precision
    # Accuracy
    print("Measure of total number of predictions the model got correct.")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy: .2f} \n")

    # Recall
    print("Percent of response values of interest captured by the model.")
    recall = recall_score(y_test, y_pred)
    print(f"Model Recall: {recall: .2f} \n")

    # Precision
    print("Measures that percentage of predicted reponse values that were correct.")
    precision = precision_score(y_test, y_pred)
    print(f"Model Precision: {precision: .2f} \n")
    
    # Classification report
    print(classification_report(y_test, y_pred))

def main():
    prepare_data_for_analysis()
    exploratory_data_analysis()
    logistic_regression()

if __name__ == "__main__":
    main()