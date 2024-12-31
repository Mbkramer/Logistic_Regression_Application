# This is a practice application logistic regression and is heavily influenced by online resources
# Inspired by a code tutorial.
# Link: https://www.kaggle.com/code/alexandreao/logistic-regression-0-98-acc-on-breast-cancer?scriptVersionId=123698399
# Using the breast cancer diagnostic dataset made available with kagle
# Link: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data?resource=download 
# Writtern by Max Kramer
# Date: 12/13/24

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

# Store diabetetes data as a global data frame
df = pd.read_csv('Datasets/breast_cancer_wisconsin_dataset.csv')

def prepare_data_for_analysis():

    # Drop column of unamed variables as well as id column
    df.drop(["id"], axis=1, inplace=True)

    # Convert diagnosis into binary target.. 1 = M or Malignant, 0 = B for Benign
    df.diagnosis = [1 if value =="M" else 0 for value in df.diagnosis] 

def exploratory_data_analysis():

    print("\nWalking through the diabetetes data frame. \n")
    print("Breast cancer dataframe head: ")
    print(df.head())

    print("\nBreast cancer dataframe info: ")
    print(df.info())

    input("\nEnter command into console to conintue..\n")
    print("Please wait for a heatmap to compare all variables within the diabetes dataframe. To continue to applying the logistic regression close the figure window. \n")

    plt.figure(figsize=(16, 12))
    sns.heatmap(df.corr(), annot=True, linewidths=2)
    plt.show()

def logistic_regression():

    print("Now that the data has been prepared, the diagnosis column will be removed from the predictor variables to train a prediction model.")
    print("In order to fit a logistc regression to the predictors data, the predictors will be scaled to ensure that they are comparably impactful.")
    print("After the predictors are scaled, both the predictors and target data will be split into train and test the model to assess how well it predicts a diagnosis. \n")
    
    y = df["diagnosis"] # Target variables are moved into a single column target df
    X = df.drop(['diagnosis'], axis=1) # Independent variables are moved into a predictors df

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