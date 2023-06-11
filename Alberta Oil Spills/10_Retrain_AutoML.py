# Databricks notebook source
# MAGIC %run ./4_MLFlow_Helpers $reset_all_data=false $catalog="hive_metastore"

# COMMAND ----------

# DBTITLE 1,Load the Features
fs = FeatureStoreClient()
features = fs.read_table(f'hive_metastore.ab_oil_spills_{initials}.morbidity_features')

# COMMAND ----------

# DBTITLE 1,Run AutoML Again
import databricks.automl
model = databricks.automl.classify(features, target_col = "is_morbid", data_dir= "dbfs:/tmp/", timeout_minutes=5) 

# COMMAND ----------

# DBTITLE 1,Register the Best Run
import mlflow
from mlflow.tracking.client import MlflowClient

client = MlflowClient()

run_id = model.best_trial.mlflow_run_id
model_name = f"dbdemos_mlops_morbidity_{initials}"
model_uri = f"runs:/{run_id}/model"

client.set_tag(run_id, key='db_table', value=f'hive_metastore.ab_oil_spills_{initials}.morbidity_features')
client.set_tag(run_id, key='demographic_vars', value='incident_number')

model_details = mlflow.register_model(model_uri, model_name)

# COMMAND ----------

# DBTITLE 1,Add Descriptions
model_version_details = client.get_model_version(name=model_name, version=model_details.version)

client.update_model_version(
  name=model_details.name,
  version=model_details.version,
  description="This model version was built using autoML and automatically getting the best model."
)

# COMMAND ----------

# DBTITLE 1,Request Transition to Staging
# Transition request to staging
staging_request = {'name': model_name, 'version': model_details.version, 'stage': 'Staging', 'archive_existing_versions': 'true'}
mlflow_call_endpoint('transition-requests/create', 'POST', json.dumps(staging_request))
