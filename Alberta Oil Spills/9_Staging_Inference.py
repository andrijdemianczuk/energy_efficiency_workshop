# Databricks notebook source
# MAGIC %md
# MAGIC # Alberta Oil Spills
# MAGIC
# MAGIC Source data set: https://data.edmonton.ca/api/views/ek45-xtjs/rows.csv?accessType=DOWNLOAD
# MAGIC <br/><br/>
# MAGIC <img src="https://static.nationalgeographic.co.uk/files/styles/image_3200/public/10-oil-sands-canada.jpg?w=800&h=300>" />
# MAGIC <br/><br/>
# MAGIC We will be building a predictor of morbidity rate (injury or death)

# COMMAND ----------

# MAGIC %md
# MAGIC ## The objective
# MAGIC In this notebook we will use the newly promoted model for some test inferencing

# COMMAND ----------

# MAGIC %run ./4_MLFlow_Helpers $reset_all_data=false $catalog="hive_metastore"

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ##Deploying the model for batch inferences
# MAGIC
# MAGIC <img style="float: right; margin-left: 20px" width="600" src="https://github.com/QuentinAmbard/databricks-demo/raw/main/retail/resources/images/churn_batch_inference.gif" />
# MAGIC
# MAGIC Now that our model is available in the Registry, we can load it to compute our inferences and save them in a table to start building dashboards.
# MAGIC
# MAGIC We will use MLFlow function to load a pyspark UDF and distribute our inference in the entire cluster. If the data is small, we can also load the model with plain python and use a pandas Dataframe.
# MAGIC
# MAGIC If you don't know how to start, Databricks can generate a batch inference notebook in just one click from the model registry !

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

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ##Deploying the model for real-time inferences
# MAGIC
# MAGIC <img style="float: right; margin-left: 20px" width="600" src="https://github.com/QuentinAmbard/databricks-demo/raw/main/retail/resources/images/churn_realtime_inference.gif" />
# MAGIC
# MAGIC Our marketing team also needs to run inferences in real-time using REST api (send a customer ID and get back the inference).
# MAGIC
# MAGIC While Feature store integration in real-time serving will come with Model Serving v2, you can deploy your Databricks Model in a single click.
# MAGIC
# MAGIC Open the Model page and click on "Serving". It'll start your model behind a REST endpoint and you can start sending your HTTP requests!

# COMMAND ----------

# MAGIC %md
# MAGIC ## Lab challenge
# MAGIC How else could we test inference of our model?

# COMMAND ----------

# DBTITLE 1,Lab Challenge Cell
#hint: Think about our original dataset and how we can hold back data. This would allow us to infer against completely 'new' data.
