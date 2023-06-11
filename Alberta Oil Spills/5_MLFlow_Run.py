# Databricks notebook source
# MAGIC %run ./4_MLFlow_Helpers $reset_all_data=false $catalog="hive_metastore"

# COMMAND ----------

# DBTITLE 1,Kick Off the AutoML Job
display_automl_churn_link(f'hive_metastore.ab_oil_spills_{initials}.morbidity_features', force_refresh=True)
