# Databricks notebook source
# MAGIC %run ../includes/configuration

# COMMAND ----------

# MAGIC %md
# MAGIC ### Setup
# MAGIC 
# MAGIC In this case we'll grab a CSV from the web, but we could also use Python or Spark to read data from databases or cloud storage.

# COMMAND ----------

# MAGIC %sh
# MAGIC # Download the CSV data file and save it in the temporary file:/tmp/ folder
# MAGIC CSV_FILE="Telco-Customer-Churn.csv"
# MAGIC rm /tmp/$CSV_FILE
# MAGIC wget https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/$CSV_FILE -P /tmp/

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load into Delta Lake

# COMMAND ----------

# MAGIC %md
# MAGIC #### Prepare the database and tables

# COMMAND ----------

# Load libraries
import shutil
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g., pd.read_csv)
from pyspark.sql.functions import col, when
from pyspark.sql.types import StructType, StructField, DoubleType, StringType, IntegerType, FloatType

# Downloaded file name
csv_file = "Telco-Customer-Churn.csv"

# Move file from the temporary folder to DBFS
driver_to_dbfs_path = f"dbfs:/home/{user}/{database_name}/{csv_file}"
dbutils.fs.cp(f"file:/tmp/{csv_file}", driver_to_dbfs_path)

# Delete the old database and tables if needed
spark.sql(f"DROP DATABASE IF EXISTS {database_name} CASCADE")

# Create database to house tables
spark.sql(f"CREATE DATABASE {database_name}")

# Delete any old Delta table files if needed (e.g., re-running this notebook with the same table paths)
shutil.rmtree(f"/dbfs{bronze_tbl_path}", ignore_errors=True)
shutil.rmtree(f"/dbfs{silver_tbl_path}", ignore_errors=True)
shutil.rmtree(f"/dbfs{telco_preds_path}", ignore_errors=True)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Read and display

# COMMAND ----------

# Define schema
schema = StructType([
  StructField('customerID', StringType()),
  StructField('gender', StringType()),
  StructField('seniorCitizen', DoubleType()),
  StructField('partner', StringType()),
  StructField('dependents', StringType()),
  StructField('tenure', DoubleType()),
  StructField('phoneService', StringType()),
  StructField('multipleLines', StringType()),
  StructField('internetService', StringType()), 
  StructField('onlineSecurity', StringType()),
  StructField('onlineBackup', StringType()),
  StructField('deviceProtection', StringType()),
  StructField('techSupport', StringType()),
  StructField('streamingTV', StringType()),
  StructField('streamingMovies', StringType()),
  StructField('contract', StringType()),
  StructField('paperlessBilling', StringType()),
  StructField('paymentMethod', StringType()),
  StructField('monthlyCharges', DoubleType()),
  StructField('totalCharges', DoubleType()),
  StructField('churnString', StringType())
  ])

# Read CSV, write to Delta and take a look
bronze_df = (spark.read.format('csv')
                       .schema(schema)
                       .option('header', 'true')
                       .load(driver_to_dbfs_path))

(bronze_df.write.format('delta')
                .mode('overwrite')
                .save(bronze_tbl_path))

display(bronze_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create bronze table

# COMMAND ----------

# Create bronze table
spark.sql(f"""
CREATE TABLE `{database_name}`.{bronze_tbl_name} USING DELTA LOCATION '{bronze_tbl_path}'
""")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
