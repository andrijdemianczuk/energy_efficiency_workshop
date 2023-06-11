# Databricks notebook source
# MAGIC %md
# MAGIC # Alberta Oil Spills
# MAGIC
# MAGIC Source data set: https://data.edmonton.ca/api/views/ek45-xtjs/rows.csv?accessType=DOWNLOAD
# MAGIC
# MAGIC We will be building a predictor of morbidity rate (injury or death)

# COMMAND ----------

# MAGIC %md
# MAGIC In this notebook we're going to preprocess and encode our data to get it ready for ML experimentation.

# COMMAND ----------

# MAGIC %run ./4_MLFlow_Helpers $reset_all_data=false $catalog="hive_metastore"

# COMMAND ----------

# DBTITLE 1,Initialize Variables
dbName = f"ab_oil_spills_{initials}"

# COMMAND ----------

# DBTITLE 1,Load the source data
cleaned_df = spark.table(f"hive_metastore.ab_oil_spills_{initials}.cleaned_oil_spills")
display(cleaned_df)
cleaned_df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepocessing Logic
# MAGIC Since we'll be assessing morbidity rate (e.g., either injury or death) we'll need to coalesce that data into a common field. By issuing a simple pyspark withColumn() function we can do this and append a new column.

# COMMAND ----------

from pyspark.sql.functions import col, when, lit

cleaned_df = cleaned_df.withColumn(
    "is_morbid",
    when((col("injury_count") > 0) | (col("fatality_count") > 0), lit(1)).otherwise(
        lit(0)
    ),
)

drop_cols = [
    "licensee_name",
    "field_centre",
    "incident_notification_date",
    "incident_date",
    "geometry_point",
    "location_1",
    "location",
    "incident_complete_date",
    "release_cleanup_date",
    "licence_number",
]

truncated_df = cleaned_df.drop(*drop_cols)
display(truncated_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Featurization Logic
# MAGIC
# MAGIC Now that we've already cleaned our data with the previous workflow to build our reduced table, we can do some one-hot encoding for our classifiers

# COMMAND ----------

# DBTITLE 1,Another option using converter / inverter
# from collections import defaultdict
# d = defaultdict(LabelEncoder)

# # Encoding the variable
# fit = data.apply(lambda x: d[x.name].fit_transform(x))

# # Inverse the encoded
# fit.apply(lambda x: d[x.name].inverse_transform(x))

# # Using the dictionary to label future data
# data.apply(lambda x: d[x.name].transform(x))

# COMMAND ----------

# DBTITLE 1,Value Encoding
from databricks.feature_store import feature_table
import pyspark.pandas as ps
import numpy as np

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


def compute_morbidity_features(data):

    # Convert to a dataframe compatible with the pandas API
    data = data.pandas_api()
    data = truncated_df.toPandas()

    # Encode all classifier features
    return data.apply(LabelEncoder().fit_transform)

# COMMAND ----------

# DBTITLE 1,Create and register the feature table in feature store
from databricks.feature_store import FeatureStoreClient

fs = FeatureStoreClient()
morbidity_features_df = compute_morbidity_features(truncated_df)
morbidity_df = spark.createDataFrame(morbidity_features_df)

try:
    # drop table if exists
    fs.drop_table(f"hive_metastore.{dbName}.dbdemos_mlops_morbidity_features")
except:
    pass
# Note: You might need to delete the FS table using the UI
churn_feature_table = fs.create_table(
    name=f"hive_metastore.{dbName}.morbidity_features",
    primary_keys="incident_number",
    schema=morbidity_df.schema,
    description="These features are derived from the cleaned_oil_spills table in the lakehouse.  I created dummy variables for the categorical columns, cleaned up their names, and added a boolean flag for whether the customer churned or not.  No aggregations were performed.",
)

fs.write_table(df=morbidity_df, name=f"{dbName}.morbidity_features", mode="overwrite")

# COMMAND ----------

truncated_df.write.format("delta").option("mergeSchema", "true").mode(
    "overwrite"
).saveAsTable(f"hive_metastore.ab_oil_spills_{initials}.trunctated_oil_spills")

# COMMAND ----------

# MAGIC %md
# MAGIC Now that we have our feature table, we can move on to the ML experiments. The Feature Store table will be used going forward as the encoded version of the truncated dataset. 
