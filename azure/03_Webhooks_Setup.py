# Databricks notebook source
# MAGIC %md
# MAGIC ## Model Registry Webhooks
# MAGIC 
# MAGIC <img src="https://github.com/RafiKurlansik/laughing-garbanzo/blob/main/step3.png?raw=true">
# MAGIC 
# MAGIC ### Supported Events
# MAGIC * Registered model created
# MAGIC * Model version created
# MAGIC * Transition request created
# MAGIC * Model version transitioned stage
# MAGIC 
# MAGIC ### Types of webhooks
# MAGIC * HTTP webhook -- send triggers to endpoints of your choosing such as slack, AWS Lambda, Azure Functions, or GCP Cloud Functions
# MAGIC * Job webhook -- trigger a job within the Databricks workspace
# MAGIC 
# MAGIC ### Use Cases
# MAGIC * Automation - automated introducing a new model to accept shadow traffic, handle deployments and lifecycle when a model is registered, etc..
# MAGIC * Model Artifact Backups - sync artifacts to a destination such as S3 or ADLS
# MAGIC * Automated Pre-checks - perform model tests when a model is registered to reduce long term technical debt
# MAGIC * SLA Tracking - Automatically measure the time from development to production including all the changes inbetween

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Webhooks
# MAGIC 
# MAGIC ___
# MAGIC 
# MAGIC <img src="https://github.com/RafiKurlansik/laughing-garbanzo/blob/main/webhooks2.png?raw=true" width = 600>

# COMMAND ----------

# MAGIC %run ../includes/configuration

# COMMAND ----------

# # Run this command only once when registering the first version of the new model
# import mlflow
# from mlflow.tracking import MlflowClient

# client = MlflowClient()

# model_uri = f"runs:/{run_id}/model"

# client.set_tag(run_id, key='db_table', value=f"{database_name}.{churn_features_tbl_name}")
# client.set_tag(run_id, key='demographic_vars', value='seniorCitizen,gender_Female')

# model_details = mlflow.register_model(model_uri, model_name)

# COMMAND ----------

# Remove all existing webhooks for our model.
# It is a cleanup to avoid duplicate webhooks for demo.
list_model_webhooks = json.dumps({"model_name": model_name})

webhooks = mlflow_call_endpoint("registry-webhooks/list", method = "GET", body = list_model_webhooks)

if len(webhooks) > 0:
  for webhook in webhooks['webhooks']:
    mlflow_call_endpoint("registry-webhooks/delete",
                       method="DELETE",
                       body = json.dumps({'id': f"{webhook['id']}"}))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Transition Request Created
# MAGIC 
# MAGIC These fire whenever a transition request is created for a model.

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Trigger Job

# COMMAND ----------

# Which model in the registry will we create a webhook for?
# The model_name variable is defined in "../includes/configuration" script

trigger_job = json.dumps({
  "model_name": model_name,
  "events": ["TRANSITION_REQUEST_CREATED"],
  "description": "Trigger the ops_validation job when a model is moved to staging.",
  "status": "ACTIVE",
  "job_spec": {
    "job_id": job_id,    # This is our 05_ops_validation notebook
    "workspace_url": host,
    "access_token": token
  }
})

mlflow_call_endpoint("registry-webhooks/create", method = "POST", body = trigger_job)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Notifications
# MAGIC 
# MAGIC Webhooks can be used to send emails, Slack messages, and more.  In this case we use Slack.  We also use `dbutils.secrets` to not expose any tokens, but the URL looks more or less like this:
# MAGIC 
# MAGIC `https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX`
# MAGIC 
# MAGIC You can read more about Slack webhooks [here](https://api.slack.com/messaging/webhooks#create_a_webhook).

# COMMAND ----------

#import urllib 
import json 

###slack_webhook = dbutils.secrets.get("demo_webhooks", "slack")

# consider REGISTERED_MODEL_CREATED to run tests and autoamtic deployments to stages 
trigger_slack = json.dumps({
  "model_name": model_name,
  "events": ["TRANSITION_REQUEST_CREATED"],
  "description": "Notify the MLOps team that a model has moved from None to Staging.",
  "status": "ACTIVE",
  "http_url_spec": {
    "url": slack_webhook
  }
})

mlflow_call_endpoint("registry-webhooks/create", method = "POST", body = trigger_slack)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Model Version Transitioned Stage
# MAGIC 
# MAGIC These fire whenever a model successfully transitions to a particular stage.

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Trigger Job

# COMMAND ----------

# Which model in the registry will we create a webhook for?
trigger_job = json.dumps({
  "model_name": model_name,
  "events": ["MODEL_VERSION_TRANSITIONED_STAGE"],
  "description": "Trigger the ops_validation job when a model is moved to staging.",
  "job_spec": {
    "job_id": job_id,
    "workspace_url": host,
    "access_token": token
  }
})

mlflow_call_endpoint("registry-webhooks/create", method = "POST", body = trigger_job)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Notifications

# COMMAND ----------

#import urllib 
import json 

trigger_slack = json.dumps({
  "model_name": model_name,
  "events": ["MODEL_VERSION_TRANSITIONED_STAGE"],
  "description": "Notify the MLOps team that a model has moved from None to Staging.",
  "http_url_spec": {
    "url": slack_webhook
  }
})

mlflow_call_endpoint("registry-webhooks/create", method = "POST", body = trigger_slack)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Manage Webhooks

# COMMAND ----------

# MAGIC %md
# MAGIC ##### List 

# COMMAND ----------

list_model_webhooks = json.dumps({"model_name": model_name})

mlflow_call_endpoint("registry-webhooks/list", method = "GET", body = list_model_webhooks)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Delete a single webhook

# COMMAND ----------

# # Delete a webhook
# mlflow_call_endpoint("registry-webhooks/delete",
#                      method="DELETE",
#                      body = json.dumps({'id': 'ff72361d77fb4e5e884a043255ca20ab'}))

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Delete all webhooks for a given model

# COMMAND ----------

# list_model_webhooks = json.dumps({"model_name": model_name})

# webhooks = mlflow_call_endpoint("registry-webhooks/list", method = "GET", body = list_model_webhooks)

# if len(webhooks) > 0:
#   for webhook in webhooks['webhooks']:
#     mlflow_call_endpoint("registry-webhooks/delete",
#                        method="DELETE",
#                        body = json.dumps({'id': f"{webhook['id']}"}))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Additional Topics & Resources
# MAGIC 
# MAGIC **Q:** Where can I find out more information on MLflow Model Registry?  
# MAGIC **A:** Check out <a href="https://mlflow.org/docs/latest/registry.html#concepts" target="_blank"> for the latest API docs available for Model Registry</a>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
