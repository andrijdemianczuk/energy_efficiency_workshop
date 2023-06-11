# Databricks notebook source
# MAGIC %md
# MAGIC # Alberta Oil Spills
# MAGIC
# MAGIC Source data set: https://data.edmonton.ca/api/views/ek45-xtjs/rows.csv?accessType=DOWNLOAD
# MAGIC
# MAGIC We will be building a predictor of morbidity rate (injury or death)

# COMMAND ----------

# MAGIC %md
# MAGIC We're going to look to see if we can build a prediction model that will help determine whether or not we are likely to have an injury or fatality due to an incident

# COMMAND ----------

# MAGIC %run ./4_MLFlow_Helpers $reset_all_data=false $catalog="hive_metastore"

# COMMAND ----------

from pyspark.sql.functions import col

# COMMAND ----------

df = spark.table(f"hive_metastore.ab_oil_spills_{initials}.raw_oil_spills")

# COMMAND ----------

# DBTITLE 1,Create a data profile to assess counts, skew etc.
display(df)

# COMMAND ----------

# DBTITLE 1,Let's get a sense of our injury / mortality incidence rate
morbidity = df.where((col("fatality_count") > 0) | (col("injury_count") > 0)).count()
print("Our morbidity rate is: " + str(round(morbidity / df.count() * 100, 4)) + "%")

# COMMAND ----------

# MAGIC %md
# MAGIC Now that we have an idea of what we want to predict, we can use this as a basis for our engineering and experimentation efforts.

# COMMAND ----------

# DBTITLE 1,Create a list of columns with reasonable non-null counts
columns = [
    "incident_meridian",
    "incident_range",
    "incident_township",
    "incident_section",
    "incident_number",
    "incident_lsd",
    "volume_released",
    "volume_recovered",
    "injury_count",
    "fatality_count",
    "licensee_name",
    "field_centre",
    "incident_notification_date",
    "incident_type",
    "incident_date",
    "geometry_point",
    "location_1",
    "location",
    "source",
    "failure_type",
    "cause_type",
    "cause_category",
    "licensee_id",
    "area_affected",
    "wildlife_livestock_affected",
    "public_affected",
    "sensitive_area",
    "release_offsite",
    "incident_complete_date",
    "strike_area",
    "substance_released",
    "volume_units",
    "release_cleanup_date",
    "environment_affected",
    "jurisdiction",
    "licence_type",
    "licence_number",
]

reduced_df = df.select(*columns)
display(reduced_df.head(5))

# COMMAND ----------

# MAGIC %md
# MAGIC There are a couple of options here: we can selectively drop / impute missing values or we can do a blind drop. Depending on the approach and precision requirements one approach might be more viable than the other. In these circumstances, experimentation for the better yield is required.

# COMMAND ----------

display(reduced_df.dropna())

# COMMAND ----------

# MAGIC %md
# MAGIC For morbid counts we drop from 363 cases down to 70.
# MAGIC <br/>
# MAGIC For total row counts, we drop from 61,587 down to 22,175.

# COMMAND ----------

# DBTITLE 1,Create a filtered list of events with no nulls
filtered_df = reduced_df.dropna()
filtered_df.count()

# COMMAND ----------

# DBTITLE 1,Let's profile our dataframe again
display(filtered_df)

# COMMAND ----------

# DBTITLE 1,Write the cleaned dataframe to a new table
filtered_df.write.format('delta').option("mergeSchema", "true").mode('overwrite').saveAsTable(f"hive_metastore.ab_oil_spills_{initials}.cleaned_oil_spills")

# COMMAND ----------

# MAGIC %md
# MAGIC This is the logical point where we'd do a handoff from the Data Engineering team to the Data Science team
