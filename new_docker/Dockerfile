FROM huggingface/transformers-pytorch-gpu:latest

ENV http_proxy "http://sgd02proxysg01.gd.rt.ru:3128"
ENV https_proxy "http://sgd02proxysg01.gd.rt.ru:3128"
WORKDIR /home/user/airflow_docker_gpu
# install requirements.txt
COPY ./docker_gpu/requirements.txt ./docker_gpu/requirements.txt
RUN python3 -m pip install -r ./docker_gpu/requirements.txt


COPY ./docker_gpu/main_inference.py ./docker_gpu/main_inference.py

COPY ./data ./data
