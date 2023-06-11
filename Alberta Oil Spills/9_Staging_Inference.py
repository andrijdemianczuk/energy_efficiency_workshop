# Databricks notebook source
# MAGIC %run ./4_MLFlow_Helpers $reset_all_data=false $catalog="hive_metastore"

# COMMAND ----------

# DBTITLE 1,Load the Model as a UDF
model = mlflow.pyfunc.spark_udf(spark, model_uri=f"models:/dbdemos_mlops_morbidity_{initials}/Staging")

# COMMAND ----------

# DBTITLE 1,Load the Features
model_features = model.metadata.get_input_schema().input_names()

# COMMAND ----------

# DBTITLE 1,Run Inferences
fs = FeatureStoreClient()
features = fs.read_table(f'hive_metastore.ab_oil_spills_{initials}.morbidity_features')

predictions = features.withColumn('morbidity_predictions', model(*model_features))
display(predictions.select("incident_number", "morbidity_predictions"))

# COMMAND ----------

# DBTITLE 1,Write to a Delta Table
predictions.write.mode("overwrite").saveAsTable(f"hive_metastore.ab_oil_spills_{initials}.morbidity_predictions")
