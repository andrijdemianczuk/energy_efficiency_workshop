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
# MAGIC ## Cluster setup
# MAGIC Create a new cluster with the following settings:
# MAGIC 1. DBR 13.x ML
# MAGIC 2. i3.xlarge or equivalent (we're not doing much heavy lifting here, so a single node is fine, as long as it's delta cache optimized)

# COMMAND ----------

# DBTITLE 1,Run the setup script
# MAGIC %run ./4_MLFlow_Helpers $reset_all_data=false $catalog="hive_metastore"

# COMMAND ----------

# DBTITLE 1,Retrieve the data from a public website
import urllib

urllib.request.urlretrieve(
    "https://data.edmonton.ca/api/views/ek45-xtjs/rows.csv", "/tmp/ab_oil_spills.csv"
)
dbutils.fs.mv("file:/tmp/ab_oil_spills.csv", "dbfs:/tmp/ab_oil_spills.csv")

# COMMAND ----------

# DBTITLE 1,Read our source data from the internet and create a dataframe
df = (
    spark.read.format("csv")
    .option("header", True)
    .option("inferSchema", True)
    .load("dbfs:/tmp/ab_oil_spills.csv")
)
df = df.withColumnRenamed(
    existing="Pipeline Outside Diameter (mm)", new="Pipeline Outside Diameter"
).withColumnRenamed(existing="Pipe Wall Thickness (mm)", new="Pipe Wall Thickness")
display(df)

# COMMAND ----------

# DBTITLE 1,Create the schema (database) if one doesn't already exist
spark.sql(f"CREATE SCHEMA IF NOT EXISTS ab_oil_spills_{initials}")

# COMMAND ----------

# DBTITLE 1,Useful function to fix the column names to strip white spaces
from pyspark.sql import DataFrame


def fix_header(df: DataFrame) -> list:
    fixed_col_list: list = []
    for col in df.columns:
        fixed_col_list.append(
            f"`{str(col).strip()}` as {str(col).strip().replace(' ','_').lower()}"
        )

    return fixed_col_list

# COMMAND ----------

# DBTITLE 1,Fix the column names
# Create a new dataframe with fixed column names
fixed_headers = fix_header(df=df)
print(fixed_headers)

# Apply to create the new dataframe
fixed_df = df.selectExpr(fixed_headers)
fixed_df.printSchema()
fixed_df.show()

# COMMAND ----------

# DBTITLE 1,Write the fixed dataframe to delta
fixed_df.write.format("delta").option("mergeSchema", "true").mode(
    "overwrite"
).saveAsTable(f"hive_metastore.ab_oil_spills_{initials}.raw_oil_spills")

# COMMAND ----------

# DBTITLE 0,LAB Challenge
# MAGIC %md
# MAGIC ## Lab Challenge
# MAGIC Can you identify how many rows are in the complete dataset?
# MAGIC
# MAGIC How many of the rows contain nulls?
# MAGIC
# MAGIC Which columns have a reasonable number nulls and which ones should we omit from our evaluation dataset?

# COMMAND ----------

# DBTITLE 1,Lab Challenge Cell
#hint: Continue working with the fixed_df dataframe using common Spark functions to help.
