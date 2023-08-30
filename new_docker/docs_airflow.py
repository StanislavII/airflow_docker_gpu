#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from datetime import date
from datetime import datetime
import posixpath

from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.operators.docker_operator import DockerOperator
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python import PythonOperator
from airflow.decorators import task
from airflow.utils.edgemodifier import Label
from airflow.sensors.filesystem import FileSensor
from airflow import macros
import docker
from docker.types import Mount 

import numpy as np
import pandas as pd
import os
import subprocess

cur_dir = os.getcwd()
today = str(date.today())

def import_file(ti):
    today = str(date.today())
    tensor = ti.xcom_pull(key = 'return_value', task_ids = 'check_gpu')
    #tensor_df = pd.DataFrame(eval(tensor))
    #tensor_df.to_csv(f"modules/simple_output_{today}.csv", index=False)

command_pull = '''cd; cd modules; smbclient '//ks.rt.ru/dfs'  -W gd -U stanislav.ilyushin%The_Beatles130299 -c 'prompt OFF;recurse ON;cd '//BEFIS2/BEFIS08/Father/docs2'; mget *' '''


#command_pull = '''cd; cd ./modules/docs2; smbget smb://ks.rt.ru/dfs/BEFIS2/BEFIS08/Father/docs2/test_document.docx -w gd -U stanislav.ilyushin%The_Beatles1302 --nonprompt '''
command_push = '''cd; smbclient  //ks.rt.ru/dfs The_Beatles130299 -W gd -U stanislav.ilyushin -c 'prompt ON;recurse ON;cd OCO/OCO_RPA/OCO_RPA12_Documents/over500_results;lcd ./modules ; put simple_output_{{macros.ds_add(ds,2)}}.csv'

 '''
command_delete = '''cd; smbclient  //ks.rt.ru/dfs The_Beatles130299 -W gd -U stanislav.ilyushin -c 'prompt ON;recurse ON;cd OCO/OCO_RPA/OCO_RPA12_Documents; del over500/*; del below500/*' '''
     
# Simple DAG
with DAG(
    "gpu_test", 
    schedule_interval="0 10 * * *", 
    start_date=datetime(2022, 1, 1), 
    catchup=False, 
    tags=['test']
) as dag:
    
    #waiting_file = FileSensor(task_id = 'waiting_file', 
    #fs_conn_id = "local_files",
    #filepath="/opt/airflow/modules/docs2/*.docx",
    #poke_interval = 30,
    #timeout = 60 * 10, 
    #mode = 'reschedule',
    #soft_fail = False)

    #bash_pull = BashOperator(task_id='bash_pull', bash_command=command_pull, do_xcom_push=True)


    check_gpu = DockerOperator(
    api_version='auto',
    docker_url='tcp://docker-proxy:2375',
    command= 'python3 ./docker_gpu/main_inference.py',
    mounts=[
        Mount(
            source='/home/user/airflow_docker_gpu/modules', 
            target='/home/user/airflow_docker_gpu/modules', 
            type='bind'
        )
    ],
    image='test_7:latest',
    task_id='check_gpu',
    do_xcom_push=True
    )
    
    #save_xcom = PythonOperator(task_id = 'save_xcom', python_callable = import_file)
    
    bash_push = BashOperator(task_id = 'bash_push', bash_command=command_push, do_xcom_push=False)
    bash_del = BashOperator(task_id = 'bash_del', bash_command=command_delete, do_xcom_push=False)
    



    # Dummy functions
    start = DummyOperator(task_id='start')
    end   = DummyOperator(task_id='end')


    # Create a simple workflow
    start >> check_gpu >> bash_push >> bash_del >> end

