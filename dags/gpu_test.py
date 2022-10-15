#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from datetime import datetime

from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.operators.docker_operator import DockerOperator
from airflow.operators.bash_operator import BashOperator
from airflow.decorators import task
from airflow.utils.edgemodifier import Label
import docker

# Simple DAG
with DAG(
    "gpu_test", 
    schedule_interval="@daily", 
    start_date=datetime(2022, 1, 1), 
    catchup=False, 
    tags=['test']
) as dag:


    check_gpu = DockerOperator(
    api_version='auto',
    docker_url='tcp://docker-proxy:2375',
    command= 'python3 main_inference.py',
    image='tensorflow/tensorflow:2.7.0-gpu',
    task_id='check_gpu'
    )



    # Dummy functions
    start = DummyOperator(task_id='start')
    end   = DummyOperator(task_id='end')


    # Create a simple workflow
    start >> check_gpu >> end

