# Databricks notebook source
# MAGIC %md
# MAGIC <img src="https://github.com/RafiKurlansik/laughing-garbanzo/blob/main/step2.png?raw=true">

# COMMAND ----------

# MAGIC %md
# MAGIC # Logistic Regression training
# MAGIC This is an auto-generated notebook. To reproduce these results, attach this notebook to the **oleg-datapalooza** cluster and rerun it.
# MAGIC - Navigate to the parent notebook [here](#notebook/3273455665514091)
# MAGIC - Compare trials in the [MLflow experiment](#mlflow/experiments/3273455665514099/s?orderByKey=metrics.%60val_f1_score%60&orderByAsc=false)
# MAGIC - Clone this notebook into your project folder by selecting **File > Clone** in the notebook toolbar.
# MAGIC 
# MAGIC Runtime Version: _8.3.x-cpu-ml-scala2.12_

# COMMAND ----------

import mlflow

# Use MLflow to track experiments
mlflow.set_experiment("/Users/oleg.kuzmin@databricks.com/databricks_automl/churn_exp_datapalooza-test")

target_col = "churn"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data

# COMMAND ----------

import os
import uuid
import shutil
import pandas as pd

from mlflow.tracking import MlflowClient

# Create temp directory to download input data from MLflow
input_temp_dir = os.path.join(os.environ["SPARK_LOCAL_DIRS"], str(uuid.uuid4())[:8])
os.makedirs(input_temp_dir)

# Download the artifact and read it into a pandas DataFrame
input_client = MlflowClient()
input_data_path = input_client.download_artifacts("876d6369afed47adaacb9d1f4e493506", "data", input_temp_dir)
df_loaded = pd.read_parquet(os.path.join(input_data_path, "training_data"))

# Delete the temp data
shutil.rmtree(input_temp_dir)

# Preview data
df_loaded.head(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preprocessors

# COMMAND ----------

transformers = []

# COMMAND ----------

# MAGIC %md
# MAGIC ### Numerical columns
# MAGIC 
# MAGIC Missing values for numerical columns are imputed with mean for consistency

# COMMAND ----------

from sklearn.impute import SimpleImputer

transformers.append(("numerical", SimpleImputer(strategy="mean"), ['contract_Month-to-month', 'contract_Oneyear', 'contract_Twoyear', 'dependents_No', 'dependents_Yes', 'deviceProtection_No', 'deviceProtection_Nointernetservice', 'deviceProtection_Yes', 'gender_Female', 'gender_Male', 'internetService_DSL', 'internetService_Fiberoptic', 'internetService_No', 'multipleLines_No', 'multipleLines_Nophoneservice', 'multipleLines_Yes', 'onlineBackup_No', 'onlineBackup_Nointernetservice', 'onlineBackup_Yes', 'onlineSecurity_No', 'onlineSecurity_Nointernetservice', 'onlineSecurity_Yes', 'paperlessBilling_No', 'paperlessBilling_Yes', 'partner_No', 'partner_Yes', 'paymentMethod_Banktransfer-automatic', 'paymentMethod_Creditcard-automatic', 'paymentMethod_Electroniccheck', 'paymentMethod_Mailedcheck', 'phoneService_No', 'phoneService_Yes', 'seniorCitizen', 'streamingMovies_No', 'streamingMovies_Nointernetservice', 'streamingMovies_Yes', 'streamingTV_No', 'streamingTV_Nointernetservice', 'streamingTV_Yes', 'techSupport_No', 'techSupport_Nointernetservice', 'techSupport_Yes', 'monthlyCharges', 'tenure', 'totalCharges']))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Categorical columns

# COMMAND ----------

# MAGIC %md
# MAGIC #### Feature hashing
# MAGIC Convert each string column into multiple numerical columns.
# MAGIC For each input string column, the number of output columns is 4096.
# MAGIC This is used for string columns with relatively many unique values.

# COMMAND ----------

from sklearn.feature_extraction import FeatureHasher
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

for feature in ['customerID']:
    hash_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(missing_values=None, strategy="constant", fill_value="")),
        (f"{feature}_hasher", FeatureHasher(n_features=4096, input_type="string"))])
    transformers.append((f"{feature}_hasher", hash_transformer, [feature]))

# COMMAND ----------

from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(transformers, remainder="passthrough", sparse_threshold=0)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature standardization
# MAGIC Scale all feature columns to be centered around zero with unit variance.

# COMMAND ----------

from sklearn.preprocessing import StandardScaler

standardizer = StandardScaler()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training - Validation Split
# MAGIC Split the input data into training and validation data

# COMMAND ----------

from sklearn.model_selection import train_test_split

split_X = df_loaded.drop([target_col], axis=1)
split_y = df_loaded[target_col]

X_train, X_val, y_train, y_val = train_test_split(split_X, split_y, random_state=442822687, stratify=split_y)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train classification model
# MAGIC - Log relevant metrics to MLflow to track runs
# MAGIC - All the runs are logged under [this MLflow experiment](#mlflow/experiments/3273455665514099/s?orderByKey=metrics.%60val_f1_score%60&orderByAsc=false)
# MAGIC - Change the model parameters and re-run the training cell to log a different trial to the MLflow experiment
# MAGIC - To view the full list of tunable hyperparameters, check the output of the cell below

# COMMAND ----------

from sklearn.linear_model import LogisticRegression

help(LogisticRegression)

# COMMAND ----------

import mlflow
import sklearn
from sklearn import set_config
from sklearn.pipeline import Pipeline

set_config(display='diagram')

sklr_classifier = LogisticRegression(
  C=7.692262065239437,
  penalty="l2",
  random_state=694904931,
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("standardizer", standardizer),
    ("classifier", sklr_classifier),
])

model

# COMMAND ----------

# Enable automatic logging of input samples, metrics, parameters, and models
mlflow.sklearn.autolog(log_input_examples=True, silent=True)

with mlflow.start_run(run_name="logistic_regression") as mlflow_run:
    model.fit(X_train, y_train)
    
    # Training metrics are logged by MLflow autologging
    # Log metrics for the validation set
    sklr_val_metrics = mlflow.sklearn.eval_and_log_metrics(model, X_val, y_val,
                                                                prefix="val_")
    display(pd.DataFrame(sklr_val_metrics, index=[0]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature importance
# MAGIC 
# MAGIC SHAP is a game-theoretic approach to explain machine learning models, providing a summary plot
# MAGIC of the relationship between features and model output. Features are ranked in descending order of
# MAGIC importance, and impact/color describe the correlation between the feature and the target variable.
# MAGIC - To reduce the computational overhead of each trial, a single example is sampled from the validation set to explain.<br />
# MAGIC   For more thorough results, increase the sample size of explanations, or provide your own examples to explain.
# MAGIC - SHAP cannot explain models using data with nulls; if your dataset has any, both the background data and
# MAGIC   examples to explain will be imputed using the mode (most frequent values). This affects the computed
# MAGIC   SHAP values, as the imputed samples may not match the actual data distribution.
# MAGIC 
# MAGIC For more information on how to read Shapley values, see the [SHAP documentation](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html).

# COMMAND ----------

from shap import KernelExplainer, summary_plot

try:
    # Sample background data for SHAP Explainer. Increase the sample size to reduce variance.
    train_sample = X_train.sample(n=min(100, len(X_train.index)))

    # Sample a single example from the validation set to explain. Increase the sample size and rerun for more thorough results.
    example = X_val.sample(n=1)

    # Use Kernel SHAP to explain feature importance on the example from the validation set.
    predict = lambda x: model.predict_proba(pd.DataFrame(x, columns=X_train.columns))
    explainer = KernelExplainer(predict, train_sample, link="logit")
    shap_values = explainer.shap_values(example, l1_reg=False)
    summary_plot(shap_values, example)
except Exception as e:
    print(f"An unexpected error occurred while plotting feature importance using SHAP: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inference
# MAGIC [The MLflow Model Registry](https://docs.databricks.com/applications/mlflow/model-registry.html) is a collaborative hub where teams can share ML models, work together from experimentation to online testing and production, integrate with approval and governance workflows, and monitor ML deployments and their performance. The snippets below show how to add the model trained in this notebook to the model registry and to retrieve it later for inference.
# MAGIC 
# MAGIC > **NOTE:** The `model_uri` for the model already trained in this notebook can be found in the cell below
# MAGIC 
# MAGIC ### Register to Model Registry
# MAGIC ```
# MAGIC model_name = "Example"
# MAGIC 
# MAGIC model_uri = f"runs:/{ mlflow_run.info.run_id }/model"
# MAGIC registered_model_version = mlflow.register_model(model_uri, model_name)
# MAGIC ```
# MAGIC 
# MAGIC ### Load from Model Registry
# MAGIC ```
# MAGIC model_name = "Example"
# MAGIC model_version = registered_model_version.version
# MAGIC 
# MAGIC model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
# MAGIC model.predict(input_X)
# MAGIC ```
# MAGIC 
# MAGIC ### Load model without registering
# MAGIC ```
# MAGIC model_uri = f"runs:/{ mlflow_run.info.run_id }/model"
# MAGIC 
# MAGIC model = mlflow.pyfunc.load_model(model_uri)
# MAGIC model.predict(input_X)
# MAGIC ```

# COMMAND ----------

# model_uri for the generated model
print(f"runs:/{ mlflow_run.info.run_id }/model")