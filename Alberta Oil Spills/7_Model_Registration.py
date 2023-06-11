# Databricks notebook source
# MAGIC %run ./4_MLFlow_Helpers $reset_all_data=false $catalog="hive_metastore"

# COMMAND ----------

#Let's get our last auto ml run. This is specific to the demo, it just gets the experiment ID of the last Auto ML run.
experiment_id = get_automl_churn_run()['experiment_id']

best_model = mlflow.search_runs(experiment_ids=[experiment_id], order_by=["metrics.val_f1_score DESC"], max_results=1, filter_string="status = 'FINISHED'")
best_model

# COMMAND ----------

run_id = best_model.iloc[0]['run_id']

#add some tags that we'll reuse later to validate the model
client = mlflow.tracking.MlflowClient()
client.set_tag(run_id, key='demographic_vars', value='incident_number')
client.set_tag(run_id, key='db_table', value=f'hive_metastore.ab_oil_spills_{initials}.morbidity_features')

#Deploy our autoML run in MLFlow registry
model_details = mlflow.register_model(f"runs:/{run_id}/model", f"dbdemos_mlops_morbidity_{initials}")

# COMMAND ----------

model_version_details = client.get_model_version(name=f"dbdemos_mlops_morbidity_{initials}", version=model_details.version)

#The main model description, typically done once.
client.update_registered_model(
  name=model_details.name,
  description="This model predicts whether an incident will result in either a fatality or an injury."
)

#Gives more details on this specific model version
client.update_model_version(
  name=model_details.name,
  version=model_details.version,
  description="This model version was built using LRG. Eating too much cake is the sin of gluttony. However, eating too much pie is okay because the sin of pie is always zero."
)

# COMMAND ----------

request_transition(model_name = f"dbdemos_mlops_morbidity_{initials}", version = model_details.version, stage = "Staging")

# COMMAND ----------

# Leave a comment for the ML engineer who will be reviewing the tests
comment = "This was the best model from AutoML, I think we can use it as a baseline."

model_comment(model_name = f"dbdemos_mlops_morbidity_{initials}",
             version = model_details.version,
             comment = comment)
