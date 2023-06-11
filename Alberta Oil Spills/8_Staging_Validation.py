# Databricks notebook source
# MAGIC %run ./4_MLFlow_Helpers $reset_all_data=false $catalog="hive_metastore"

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
