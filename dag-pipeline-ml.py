# https://medium.datadriveninvestor.com/machine-learning-orchestration-using-apache-airflow-beginner-level-e4939492568c

from datetime import timedelta
# The DAG object; we'll need this to instantiate a DAG
from airflow import DAG

from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago

import sys
sys.path.append("src")
from logger import logging

from python_functions import download_dataset_fn
from python_functions import data_processing_fn
from python_functions import ml_training_RandomForest_fn
from python_functions import ml_training_Logisitic_fn
from python_functions import identify_best_model_fn

args={
    'owner' : 'airflow',
    'retries': 1,
    'start_date':days_ago(1)  #1 means yesterday
}


with DAG(
    dag_id='airflow_ml_pipeline', ## Name of DAG run
    default_args=args,
    description='ML pipeline',
    schedule = None,  
 ) as dag:

 # Task 1 - Just a simple print statement
 dummy_task = EmptyOperator(task_id='Starting_the_process', retries=2)  

# Task 2 - Download the dataset
 task_extract_data = PythonOperator(
 task_id='download_dataset',
 python_callable=download_dataset_fn
 )

 # Task 3 - Transform the data
 task_process_data = PythonOperator(
 task_id='data_processing',
 python_callable=data_processing_fn
 )

 # Task 4_A - Train a ML Model using Random Forest
 task_train_RF_model = PythonOperator(
 task_id='ml_training_RandomForest',
 python_callable=ml_training_RandomForest_fn
 )

 # Task 4_B -Train a ML Model using Logistic Regression
 task_train_LR_model = PythonOperator(
 task_id='ml_training_Logisitic',
 python_callable=ml_training_Logisitic_fn
 )

 # Task 5 -Identify the best model of the two
 task_identify_best_model = PythonOperator(
 task_id='identify_best_model',
 python_callable=identify_best_model_fn
 )


# Define the workflow process
dummy_task  >> task_extract_data >> task_process_data >> [task_train_RF_model,task_train_LR_model] >> task_identify_best_model
