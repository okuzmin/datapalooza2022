# Databricks notebook source
# MAGIC %md
# MAGIC ### Managing the model lifecycle with Model Registry
# MAGIC 
# MAGIC <img src="https://github.com/RafiKurlansik/laughing-garbanzo/blob/main/step4.png?raw=true">
# MAGIC 
# MAGIC One of the primary challenges among data scientists and ML engineers is the absence of a central repository for models, their versions, and the means to manage them throughout their lifecycle.  
# MAGIC 
# MAGIC [The MLflow Model Registry](https://docs.databricks.com/applications/mlflow/model-registry.html) addresses this challenge and enables members of the data team to:
# MAGIC <br><br>
# MAGIC * **Discover** registered models, current stage in model development, experiment runs, and associated code with a registered model
# MAGIC * **Transition** models to different stages of their lifecycle
# MAGIC * **Deploy** different versions of a registered model in different stages, offering MLOps engineers ability to deploy and conduct testing of different model versions
# MAGIC * **Test** models in an automated fashion
# MAGIC * **Document** models throughout their lifecycle
# MAGIC * **Secure** access and permission for model registrations, transitions or modifications
# MAGIC 
# MAGIC <img src="https://databricks.com/wp-content/uploads/2020/04/databricks-adds-access-control-to-mlflow-model-registry_01.jpg">

# COMMAND ----------

# MAGIC %run ../includes/configuration

# COMMAND ----------

# MAGIC %md
# MAGIC ### How to Use the Model Registry
# MAGIC Typically, data scientists who use MLflow will conduct many experiments, each with a number of runs that track and log metrics and parameters. During the course of this development cycle, they will select the best run within an experiment and register its model with the registry.  Think of this as **committing** the model to the registry, much as you would commit code to a version control system.  
# MAGIC 
# MAGIC The registry defines several model stages: `None`, `Staging`, `Production`, and `Archived`. Each stage has a unique meaning. For example, `Staging` is meant for model testing, while `Production` is for models that have completed the testing or review processes and have been deployed to applications. 
# MAGIC 
# MAGIC Users with appropriate permissions can transition models between stages.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Promote to Registry
# MAGIC ```
# MAGIC model_name = "Example"
# MAGIC 
# MAGIC model_uri = f"runs:/{ mlflow_run.info.run_id }/model"
# MAGIC registered_model_version = mlflow.register_model(model_uri, model_name)
# MAGIC ```

# COMMAND ----------

# This command will create a new version of an existing model
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

model_uri = f"runs:/{run_id}/model"

client.set_tag(run_id, key='db_table', value=f"{database_name}.{churn_features_tbl_name}")
client.set_tag(run_id, key='demographic_vars', value='seniorCitizen,gender_Female')

model_details = mlflow.register_model(model_uri, model_name)

# COMMAND ----------

# MAGIC %md
# MAGIC At this point the model will be in `None` stage.  Let's update the description before moving it to `Staging`.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Update Description

# COMMAND ----------

from mlflow.tracking.client import MlflowClient

client = MlflowClient()
model_version_details = client.get_model_version(name=model_name, version=model_details.version)

client.update_registered_model(
  name=model_details.name,
  description=f"This model predicts whether a customer will churn using features from the {database_name} database.  It is used to update the Telco Churn Dashboard in SQL Analytics."
)

client.update_model_version(
  name=model_details.name,
  version=model_details.version,
  description="This model version was built using sklearn's LogisticRegression - or maybe not :)"
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Request Transition to Staging
# MAGIC 
# MAGIC <img src="https://github.com/RafiKurlansik/laughing-garbanzo/blob/main/webhooks2.png?raw=true" width = 800>

# COMMAND ----------

# MAGIC %run ./helpers/registry_helpers

# COMMAND ----------

# Transition request to staging
staging_request = {'name': model_name,
                   'version': model_details.version,
                   'stage': 'Staging',
                   'archive_existing_versions': 'true'}

mlflow_call_endpoint('transition-requests/create', 'POST', json.dumps(staging_request))

# COMMAND ----------

# Leave a comment for the ML engineer who will be reviewing the tests
comment = "This was the best model from AutoML, I think we can use it as a baseline."
comment_body = {'name': model_name, 'version': model_details.version, 'comment': comment}
mlflow_call_endpoint('comments/create', 'POST', json.dumps(comment_body))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
