# Databricks notebook source
import mlflow
import urllib
import json

class HttpClient:
  def __init__(self, base_url, token):
    self.base_url = base_url
    self.token = token
  
  def createWebhook(self, request):
    return self._post('api/2.0/mlflow/registry-webhooks/create', request)

  def updateWebhook(self, request):
    return self._patch('api/2.0/mlflow/registry-webhooks/update', request)

  def listWebhooks(self, request):
    return self._get('api/2.0/mlflow/registry-webhooks/list', request)

  def deleteWebhook(self, request):
    return self._delete('api/2.0/mlflow/registry-webhooks/delete', request)
    
  def _get(self, uri, params):
    data = urllib.parse.urlencode(params)
    url = f'{self.base_url}/{uri}/?{data}'
    headers = { 'Authorization': f'Bearer {self.token}'}

    req = urllib.request.Request(url, headers=headers)
    response = urllib.request.urlopen(req)
    return json.load(response)

  def _post(self, uri, body):
    json_body = json.dumps(body)
    json_bytes = json_body.encode('utf-8')
    headers = { 'Authorization': f'Bearer {self.token}'}

    url = f'{self.base_url}/{uri}'
    req = urllib.request.Request(url, data=json_bytes, headers=headers)
    response = urllib.request.urlopen(req)
    return json.load(response)

  def _patch(self, uri, body):
    json_body = json.dumps(body)
    json_bytes = json_body.encode('utf-8')
    headers = { 'Authorization': f'Bearer {self.token}'}

    url = f'{self.base_url}/{uri}'
    req = urllib.request.Request(url, data=json_bytes, headers=headers, method='PATCH')
    response = urllib.request.urlopen(req)
    return json.load(response)

  def _delete(self, uri, body):
    json_body = json.dumps(body)
    json_bytes = json_body.encode('utf-8')
    headers = { 'Authorization': f'Bearer {self.token}'}

    url = f'{self.base_url}/{uri}'
    req = urllib.request.Request(url, data=json_bytes, headers=headers, method='DELETE')
    response = urllib.request.urlopen(req)
    return json.load(response)

# COMMAND ----------

## SETUP 

### STEP 1:Databricks access token 
TOKEN = 'dapiXXXXXXXXXXXXXXXXXXXXXXXX'
URL = 'https://XXXXXXXXXX.cloud.databricks.com'
## Step 2: Instantiate a new HttpClient
httpClient = HttpClient(URL, TOKEN)

# COMMAND ----------

model_name = "telco_churn_demo_model"

# COMMAND ----------

# Create a Job webhook to run the validation job
httpClient.createWebhook({
  "model_name": model_name,
  "events": ["TRANSITION_REQUEST_CREATED"],
  "status": "ACTIVE",
  "job_spec": {
    "job_id": 3110,
    "workspace_url": URL,
    "access_token": TOKEN
  }
})

# COMMAND ----------

# Create a webhook that will alert us about transition requests that we create
httpClient.createWebhook({
  "model_name": model_name,
  "events": ["TRANSITION_REQUEST_CREATED", "MODEL_VERSION_CREATED"],
  "status": "ACTIVE",
  "http_url_spec": {"url": "https://hooks.slack.com/services/XXXXXXXXXXXXXXXX"}
})

# COMMAND ----------

# Create a Job webhook to run the batch inference job
httpClient.createWebhook({
  "model_name": model_name,
  "events": ["MODEL_VERSION_TRANSITIONED_STAGE"],
  "status": "ACTIVE",
  "job_spec": {
    "job_id": 553,
    "workspace_url": URL,
    "access_token": TOKEN
  }
})

# COMMAND ----------

webhooks = httpClient.listWebhooks({
  "events": ["MODEL_VERSION_CREATED", "MODEL_VERSION_TRANSITIONED_STAGE", "TRANSITION_REQUEST_CREATED"],
})
webhooks
