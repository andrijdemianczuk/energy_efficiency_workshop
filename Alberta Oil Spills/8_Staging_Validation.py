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
# MAGIC Now that we've registered our model in staging, we can evaluate it programmatically. This is similar to a DevOps practice where we assess whether all required criteria are met before we sign off and promote the model to production.

# COMMAND ----------

# MAGIC %run ./4_MLFlow_Helpers $reset_all_data=false $catalog="hive_metastore"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fetch Model information
# MAGIC
# MAGIC Remember how webhooks can send data from one webservice to another?  With MLflow webhooks we send data about a model, and in the following cell we fetch that data to know which model is meant to be tested. 
# MAGIC
# MAGIC This is be done getting the `event_message` received by MLFlow webhook: `dbutils.widgets.get('event_message')`
# MAGIC
# MAGIC To keep things simple we use a helper function `fetch_webhook_data`, the details of which are found in the _API_Helpers_ notebook.  

# COMMAND ----------

# DBTITLE 1,Get Model Info
# Get the model in transition, its name and version from the metadata received by the webhook
model_name, model_version = fetch_webhook_data()

client = MlflowClient()
model_name = f"dbdemos_mlops_morbidity_{initials}"
model_details = client.get_model_version(model_name, model_version)
run_info = client.get_run(run_id=model_details.run_id)

# COMMAND ----------

# DBTITLE 1,Validate Prediction
from databricks.feature_store import FeatureStoreClient

fs = FeatureStoreClient()

# Read from feature store
data_source = run_info.data.tags["db_table"]
features = fs.read_table(data_source)

# Load model as a Spark UDF
model_uri = f"models:/{model_name}/{model_version}"
loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=model_uri)

# Select the feature table cols by model input schema
input_column_names = loaded_model.metadata.get_input_schema().input_names()

# Predict on a Spark DataFrame
try:
    display(features.withColumn("predictions", loaded_model(*input_column_names)))
    client.set_model_version_tag(
        name=model_name, version=model_version, key="predicts", value=1
    )
except Exception:
    print("Unable to predict on features.")
    client.set_model_version_tag(
        name=model_name, version=model_version, key="predicts", value=0
    )
    pass

# COMMAND ----------

# MAGIC %md
# MAGIC #### Signature check
# MAGIC
# MAGIC When working with ML models you often need to know some basic functional properties of the model at hand, such as “What inputs does it expect?” and “What output does it produce?”.  The model **signature** defines the schema of a model’s inputs and outputs. Model inputs and outputs can be either column-based or tensor-based. 
# MAGIC
# MAGIC See [here](https://mlflow.org/docs/latest/models.html#signature-enforcement) for more details.

# COMMAND ----------

# DBTITLE 1,Signature Check
if not loaded_model.metadata.signature:
  print("This model version is missing a signature.  Please push a new version with a signature!  See https://mlflow.org/docs/latest/models.html#model-metadata for more details.")
  client.set_model_version_tag(name=model_name, version=model_version, key="has_signature", value=0)
else:
  client.set_model_version_tag(name=model_name, version=model_version, key="has_signature", value=1)

# COMMAND ----------

# DBTITLE 1,Demographic Accuracy
import numpy as np

features = features.withColumn(
    "predictions", loaded_model(*input_column_names)
).toPandas()
features["accurate"] = np.where(features.is_morbid == features.predictions, 1, 0)

# Check run tags for demographic columns and accuracy in each segment
try:
    demographics = run_info.data.tags["demographic_vars"].split(",")
    slices = features.groupby(demographics).accurate.agg(
        acc="sum", obs=lambda x: len(x), pct_acc=lambda x: sum(x) / len(x)
    )

    # Threshold for passing on demographics is 55%
    demo_test = "pass" if slices["pct_acc"].any() > 0.55 else "fail"

    # Set tags in registry
    client.set_model_version_tag(
        name=model_name, version=model_version, key="demo_test", value=demo_test
    )

    print(slices)
except KeyError:
    print("KeyError: No demographics_vars tagged with this model version.")
    client.set_model_version_tag(
        name=model_name, version=model_version, key="demo_test", value="none"
    )
    pass

# COMMAND ----------

# DBTITLE 1,Description Check
# If there's no description or an insufficient number of charaters, tag accordingly
if not model_details.description:
    client.set_model_version_tag(
        name=model_name, version=model_version, key="has_description", value=0
    )
    print("Did you forget to add a description?")
elif not len(model_details.description) > 20:
    client.set_model_version_tag(
        name=model_name, version=model_version, key="has_description", value=0
    )
    print(
        "Your description is too basic, sorry.  Please resubmit with more detail (40 char min)."
    )
else:
    client.set_model_version_tag(
        name=model_name, version=model_version, key="has_description", value=1
    )

# COMMAND ----------

# DBTITLE 1,Artifacts Check
import os

# Create local directory
local_dir = "/tmp/model_artifacts"
if not os.path.exists(local_dir):
    os.mkdir(local_dir)

# Download artifacts from tracking server - no need to specify DBFS path here
local_path = client.download_artifacts(run_info.info.run_id, "", local_dir)

# Tag model version as possessing artifacts or not
if not os.listdir(local_path):
    client.set_model_version_tag(
        name=model_name, version=model_version, key="has_artifacts", value=0
    )
    print(
        "There are no artifacts associated with this model.  Please include some data visualization or data profiling.  MLflow supports HTML, .png, and more."
    )
else:
    client.set_model_version_tag(
        name=model_name, version=model_version, key="has_artifacts", value=1
    )
    print("Artifacts downloaded in: {}".format(local_path))
    print("Artifacts: {}".format(os.listdir(local_path)))

# COMMAND ----------

# DBTITLE 1,Results Summary
results = client.get_model_version(model_name, model_version)
results.tags

# COMMAND ----------

# DBTITLE 1,Move to Staging or Archived
# If any checks failed, reject and move to Archived
if "0" in results or "fail" in results:
    print("Rejecting transition...")
    reject_transition(
        model_name,
        model_version,
        stage="Staging",
        comment="Tests failed, moving to archived.  Check the tags or the job run to see what happened.",
    )

else:
    print("Accepting transition...")
    accept_transition(
        model_name,
        model_version,
        stage="Staging",
        comment="All tests passed!  Moving to staging.",
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Lab challenge
# MAGIC In this example, we verify a few test vectors - do artifacts exist, does the model have a description, a proper signature and does it do what it needs to do. What other testing criteria would be useful on a circumstantial basis?

# COMMAND ----------

# DBTITLE 1,Lab Challenge Cell
#hint: Try altering the logic that either promotes the model or archives it based on data quality checks
