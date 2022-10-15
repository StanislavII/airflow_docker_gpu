FROM tensorflow/tensorflow:2.7.0-gpu

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libfreetype6-dev \
        pkg-config \
        python \
        python3 \
        python3-dev \
        python3-pip \
        python3-setuptools \
        software-properties-common \
        unzip \
        nano \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# install requirements.txt
COPY requirements.txt requirements.txt
RUN python3 -m pip install -r requirements.txt


COPY ./main_inference.py main_inference.py