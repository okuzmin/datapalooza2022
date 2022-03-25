# Databricks notebook source
# MAGIC %md
# MAGIC **Define database name, Delta table names and paths**

# COMMAND ----------

user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply("user")
database_name = "telco_churn_demo"

# Paths of Delta tables
bronze_tbl_path = f"/home/{user}/{database_name}/bronze/"
silver_tbl_path = f"/home/{user}/{database_name}/silver/"
automl_tbl_path = f"/home/{user}/{database_name}/automl-silver/"
telco_preds_path = f"/home/{user}/{database_name}/preds/"

# Names of Delta tables
bronze_tbl_name = "bronze_customers"
silver_tbl_name = "silver_customers"
automl_tbl_name = "gold_customers"
telco_preds_tbl_name = "telco_preds"
churn_features_tbl_name = "churn_features_datapalooza"

# Name of the ML model for the MLFlow registry
model_name = "telco_churn_demo_model"

# COMMAND ----------

# MAGIC %md
# MAGIC **Define additional variables**

# COMMAND ----------

# TODO: change this run_id as needed.
# The run_id and model_uri are available at the bottom of the cloned auto-generated model notebook (e.g., 02_AutoML_Baseline_Cloned)
run_id = "69d01284e5a842d3944b2d0266c2e62d"

# TODO: change to the Job ID for the job you configured
#job_id = "438594"
job_id = "523361"

# TODO: change to the Slack webhook you configured
slack_url = "https://hooks.slack.com/services"
slack_webhook = f"{slack_url}/T02U0R0H1GF/B02UVVCPJGM/dIWP1RBpyheDfDsDqc0Yho0V"

# COMMAND ----------

# MAGIC %md
# MAGIC **Import utility functions**

# COMMAND ----------

# MAGIC %run ./utilities
