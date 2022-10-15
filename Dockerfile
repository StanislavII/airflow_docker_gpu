FROM apache/airflow:2.3.0


COPY requirements.txt requirements.txt
RUN python3 -m pip install -r requirements.txt
