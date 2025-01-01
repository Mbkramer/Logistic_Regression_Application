# Loan approval classification regression use to predict loan approval decision
# Using loan approval classification dataset made available with kagle
# Link: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data?resource=download 
# Writtern by Max Kramer
# Date: 12/31/25

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

df = pd.read_csv('Datasets/loan_data.csv')

def prepare_data_for_analysis():

    print(df.info(), end="\n")
    print(df.head())

    df.rename(columns={'person_education': 'person_is_college_educated'}, inplace=True)
    df.drop(columns={'loan_intent'}, inplace=True)

    df.person_gender = [1 if value == 'female' else 0 for value in df.person_gender]
    df.person_is_college_educated = [0 if value == 'High School' else 1 for value in df.person_is_college_educated]
    df.person_home_ownership = [0 if value == 'RENT' else 1 for value in df.person_home_ownership]
    df.previous_loan_defaults_on_file = [1 if value == 'Yes' else 0 for value in df.previous_loan_defaults_on_file]

    print(df.info(), end="\n")
    print(df.head())

def exploratory_data_analysis():

    plt.figure(figsize=(16, 12))
    sns.heatmap(df.corr(), annot=True, linewidths=2)
    plt.show()

def logistic_regression():
    
    y = df["loan_status"] # Target variables are moved into a single column target df
    X = df.drop(['loan_status'], axis=1) # Independent variables are moved into a predictors df

    # Normalize units.. lots of variance in data so that variables are comparably impactful
    scaler = StandardScaler()

    # Fit and transform
    X_scaled = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=16)
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

if __name__=="__main__":
    main()
