FROM apache/airflow:3.1.0

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

ARG AIRFLOW_VERSION=3.1.0
ARG PYTHON_VERSION=3.12

COPY requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir \
    --user \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt" \
    -r /tmp/requirements.txt
