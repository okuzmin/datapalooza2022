# Databricks notebook source
# # Boilerplate utility functions.
# # Use these to create the actual useful functions for your demo.
# from pyspark.sql.session import SparkSession
# from urllib.request import urlretrieve
# import time

# BASE_URL = "https://foo/bar/"


# def retrieve_data(year: int, month: int, raw_path: str) -> bool:
#     file, dbfsPath, tempPath = _generate_file_handles(year, month, raw_path)
#     uri = BASE_URL + file

#     urlretrieve(uri, file)
#     dbutils.fs.mv(tempPath, dbfsPath)
#     return True


# def _generate_file_handles(year: int, month: int, raw_path: str):
#     file = f"foo_bar_{year}_{month}.json"

#     dbfsPath = raw_path
#     dbfsPath += file

#     tempPath = "file:/tmp/" + file

#     return file, dbfsPath, driverPath

# COMMAND ----------

# Webhook client functions.
# used in 03_Webhooks_Setup notebook.

import mlflow
from mlflow.utils.rest_utils import http_request
import json

def client():
  return mlflow.tracking.client.MlflowClient()

host_creds = client()._tracking_client.store.get_host_creds()
host = host_creds.host
token = host_creds.token

def mlflow_call_endpoint(endpoint, method, body='{}'):
  if method == 'GET':
      response = http_request(
          host_creds=host_creds, endpoint="/api/2.0/mlflow/{}".format(endpoint), method=method, params=json.loads(body))
  else:
      response = http_request(
          host_creds=host_creds, endpoint="/api/2.0/mlflow/{}".format(endpoint), method=method, json=json.loads(body))
  return response.json()
