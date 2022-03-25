# Databricks notebook source
# MAGIC %md
# MAGIC ## Churn Prediction Feature Engineering
# MAGIC 
# MAGIC <img src="https://github.com/RafiKurlansik/laughing-garbanzo/blob/main/step1.png?raw=true">

# COMMAND ----------

# MAGIC %run ../includes/configuration

# COMMAND ----------

# MAGIC %md
# MAGIC ## Featurization Logic
# MAGIC 
# MAGIC This is a fairly clean dataset so we'll just do some one-hot encoding and clean up the column names afterward.

# COMMAND ----------

# Read the bronze table into Spark
telcoDF = spark.table(f"{database_name}.{bronze_tbl_name}")

display(telcoDF)

# COMMAND ----------

# MAGIC %md
# MAGIC (**Databricks Runtime 7.3 through 9.1**): use `import databricks.koalas as ps` to scale my teammates' single-node `pandas` code .
# MAGIC 
# MAGIC Starting with **Databricks Runtime 10.0**, use `import pyspark.pandas as ps` instead (distributed Pandas API on Spark is available in Apache Spark 3.2)

# COMMAND ----------

###from databricks.feature_store import feature_table
#import databricks.koalas as ps
import pyspark.pandas as ps

def compute_churn_features(data):
  
  # Convert to koalas (DBR 7.3 - 9.1)
  #data = data.to_koalas()
  
  # Convert to pandas (DBR 10.0+)
  data = ps.DataFrame(data)
  
  # OHE
  data = ps.get_dummies(data, 
                        columns=['gender', 'partner', 'dependents',
                                 'phoneService', 'multipleLines', 'internetService',
                                 'onlineSecurity', 'onlineBackup', 'deviceProtection',
                                 'techSupport', 'streamingTV', 'streamingMovies',
                                 'contract', 'paperlessBilling', 'paymentMethod'],
                        dtype = 'int64')
  
  # Convert label to int and rename column
  data['churnString'] = data['churnString'].map({'Yes': 1, 'No': 0})
  data = data.astype({'churnString': 'int32'})
  data = data.rename(columns = {'churnString': 'churn'})
  
  # Clean up column names
  data.columns = data.columns.str.replace(' ', '', regex=False)
  data.columns = data.columns.str.replace('(', '-', regex=False)
  data.columns = data.columns.str.replace(')', '', regex=False)
  
  # Drop missing values
  data = data.dropna()
  
  return data

# COMMAND ----------

# MAGIC %md
# MAGIC ### Compute and write features to the Feature Store

# COMMAND ----------

from databricks.feature_store import FeatureStoreClient

fs = FeatureStoreClient()

churn_features_df = compute_churn_features(telcoDF)

# TODO: delete_table() method is not available yet, will add it to this demo when available.
# You can delete an existing feature table from the Feature Store UI now.

# for DBRs before 10.2 ML, use fs.create_feature_table() method instead of fs.create_table()
churn_feature_table = fs.create_table(
                         name=f"{database_name}.{churn_features_tbl_name}",
                         primary_keys="customerID",
                         schema=churn_features_df.spark.schema(),
                         description=f"These features are derived from the {database_name}.{bronze_tbl_name} table in the lakehouse.  I created dummy variables (one-hot encoded) for the categorical columns, cleaned up their names, and added a boolean flag for whether the customer churned or not.  No aggregations were performed."
)

(fs.write_table(
                df=churn_features_df.to_spark(),
                name=f"{database_name}.{churn_features_tbl_name}",
                mode='overwrite')
)

# COMMAND ----------

# MAGIC %md
# MAGIC As an alternative to using the Feature Store, we could always write to a Delta table:

# COMMAND ----------

# # Write out silver-level data to Delta lake
# trainingDF = spark.createDataFrame(training_df)

# trainingDF.write.format('delta').mode('overwrite').save(silver_tbl_path)

# # Create silver table
# spark.sql('''
#   CREATE TABLE `{}`.{}
#   USING DELTA 
#   LOCATION '{}'
#   '''.format(database_name,silver_tbl_name,silver_tbl_path))

# # Drop customer ID for AutoML
# automlDF = trainingDF.drop('customerID')

# # Write out silver-level data to Delta lake
# automlDF.write.format('delta').mode('overwrite').save(automl_tbl_path)

# # Create silver table
# _ = spark.sql('''
#   CREATE TABLE `{}`.{}
#   USING DELTA 
#   LOCATION '{}'
#   '''.format(database_name,automl_tbl_name,automl_tbl_path))
