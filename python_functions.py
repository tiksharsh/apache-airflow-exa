# https://medium.datadriveninvestor.com/machine-learning-orchestration-using-apache-airflow-beginner-level-e4939492568c
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import sys
import pandas as pd
import numpy as np 
import os

sys.path.append("src")

from logger import logging

# https://medium.datadriveninvestor.com/machine-learning-orchestration-using-apache-airflow-beginner-level-e4939492568c 

def download_dataset_fn():
    
    iris = load_iris()
    iris = pd.DataFrame(
    data = np.c_[iris['data'], iris['target']],
    columns = iris['feature_names'] + ['target'])
    logging.info("Downloaded the iris data set")
    pd.DataFrame(iris).to_csv("data/iris_dataset.csv")

def data_processing_fn():
    train_df_path: str = os.path.join('data',"iris_dataset.csv")

    final = pd.read_csv(train_df_path,index_col=0)
    cols = ["sepal length (cm)","sepal width (cm)","petal length (cm)","petal width (cm)"]
    final[cols] = final[cols].fillna(final[cols].mean())
    logging.info("Data processing done & saved the data in clean-iris-dataset")
    final.to_csv("data/clean_iris_dataset.csv")

def ml_training_RandomForest_fn(**kwargs):

    final = pd.read_csv("data/clean_iris_dataset.csv",index_col=0)
    X_train, X_test, y_train, y_test = train_test_split(final.iloc[:,0:4],final.iloc[:,-1], test_size=0.3)
    clf = RandomForestClassifier(n_estimators = 100)  
    clf.fit(X_train, y_train)
    logging.info("Model training using Random Forest")
    # performing predictions on the test dataset
    y_pred = clf.predict(X_test)
     
    # using metrics module for accuracy calculation
    print("ACCURACY OF THE MODEL: ", accuracy_score(y_test, y_pred))
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Accuracy of the model: {accuracy}")

    kwargs['ti'].xcom_push(key='model_accuracy', value=acc )


def ml_training_Logisitic_fn(**kwargs):

    final = pd.read_csv("data/clean_iris_dataset.csv",index_col=0)
    X_train, X_test, y_train, y_test = train_test_split(final.iloc[:,0:4],final.iloc[:,-1], test_size=0.3)
    logistic_regression = LogisticRegression(multi_class="ovr")
    lr = logistic_regression.fit(X_train, y_train)
    logging.info("Model training using Logistic Regression")
    y_pred = lr.predict(X_test)
    
    print("ACCURACY OF THE MODEL: ", accuracy_score(y_test, y_pred))
    acc = accuracy_score(y_test, y_pred)
    logging.info(f"Accuracy of the model: {acc}")
    kwargs['ti'].xcom_push(key='model_accuracy', value=acc )


def identify_best_model_fn(**kwargs):
    ti = kwargs['ti']
    fetched_accuracies = ti.xcom_pull(key='model_accuracy', task_ids=['ml_training_RandomForest', 'ml_training_Logisitic'])
    print(f'choose best model: {fetched_accuracies}')
    logging.info(f'choosen best model: {fetched_accuracies}')