#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
import requests
import posixpath
import datetime
from google.protobuf.json_format import MessageToJson
from proto_structs import offers_pb2
from typing import List
import pickle
import datetime
import numpy as np
import pandas as pd
import pathlib
import airflow
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator

today = str(datetime.date.today())

def get_url(city, shop, page_num):
    url = f"https://squark.edadeal.ru/web/search/offers?count=50&locality={city}&page={page_num}&retailer={shop}"
    return url


def check_url_code(city: str, shop: str, page_num: int):
    url = get_url(city, shop, page_num)
    return requests.get(url, allow_redirects=True).status_code


def parse_page(city: str, shop: str, page_num: int):
    """
    :param city: location of the shop
    :param shop: shop name
    :param page_num: parsed page number
    :return: None
    """

    url = get_url(city, shop, page_num)
    data = requests.get(url, allow_redirects=True)  # data.content is a protobuf message

    offers = offers_pb2.Offers()  # protobuf structure
    offers.ParseFromString(data.content)  # parse binary data
    products: str = MessageToJson(offers)  # convert protobuf message to json
    products = json.loads(products)
    data = []
    for prod in products['offer']:
        prod['city'] = city
        prod['shop'] = shop
        prod['page'] = page_num
        prod['processed_date'] = datetime.date.today().strftime('%Y-%m-%d')
        data.append(prod)
    return data


def parse_shop(city: str = 'moskva', shop: str = '5ka', page_range: int = 100, skip_errors=False):
    shop_data = []
    for page in range(page_range):
        code = check_url_code(city, shop, page_num=page)
        if code == 200:
            data = parse_page(city, shop, page_num=page)
            shop_data.extend(data)
        elif code == 204:
            if page > 1:
                print(f'No more products in {shop}')
            else:
                print(f'No data for {shop}')
            return shop_data
        else:
            if skip_errors:
                print(f'Unexpected code {code} for {shop}')
                return shop_data
            else:
                raise ValueError(f'Unexpected code {code} for {shop}')
    return shop_data


def parse_shop_list(city: str, shop_list: List[str], page_range=100):
    shop_dict = {}
    for shop in shop_list:
        shop_dict[shop] = parse_shop(city=city, shop=shop, page_range=page_range)
    return shop_dict
    
shop_list = ['5ka', 'magnit-univer', 'perekrestok', 'dixy',
                 'lenta-super', 'vkusvill_offline', 'mgnl', 'azbuka_vkusa']
                 
def final(shop_list, ds):
   
       
   home_path = f"data/history_prices/ds={ds}"
   
   pathlib.Path(home_path).mkdir(parents=True, exist_ok=True)

   data = parse_shop_list(city='moskva', shop_list=shop_list, page_range=100)
   
   with open(posixpath.join(home_path, "prices.pickle"), "wb") as handle:
       pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

   t_pd = pd.DataFrame(data)
   t_pd.to_csv(posixpath.join(home_path, "csv_frame.csv"))

dag = DAG(dag_id = 'parser_dag',
start_date = airflow.utils.dates.days_ago(7),
schedule_interval = "@daily")

python_load = PythonOperator(task_id = "parser_python",
python_callable = final,
op_kwargs = { "shop_list": shop_list
},
provide_context=True,
 dag = dag)

  
task_second = PostgresOperator(
task_id='create_postgres_table',
postgres_conn_id='postgre_sql',
sql="""
create table if not exists edadil (
dt date,
name character varying,
price_before numeric(8,2),
price_after numeric(8,2),
amount numeric(8,2),
amout_unit character varying,
discount numeric(8,2),
start_date character varying,
end_date character varying,
 city character varying,
 shop character varying,
 page integer,
 processed_date character varying,
primary key (dt, name)
)
""", dag = dag
)

task_load_data = BashOperator(
task_id='load_sql_data',
bash_command=(
'psql -d for_project -U airflow -c "'
'COPY edadil(dt, name, price_before, price_after, amount, amount_unit, discount, start_date, end_date, city, shop, page, precessed_date) '
"FROM f'data/history_prices/ds=2022-10-16/csv_frame.csv' "
"DELIMITER ',' "
'CSV HEADER"'
),
dag = dag)
notify = BashOperator(task_id = "notify",
bash_command = 'echo "Hello!"', dag = dag)

python_load >>  task_second >> task_load_data >> notify



