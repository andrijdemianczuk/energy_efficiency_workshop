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
# MAGIC
# MAGIC In this notebook, we will be using our feature table that we previously registed in feature store to initiate an AutoML experiment run.

# COMMAND ----------

# MAGIC %run ./4_MLFlow_Helpers $reset_all_data=false $catalog="hive_metastore"

# COMMAND ----------

# DBTITLE 1,Kick Off the AutoML Job
display_automl_churn_link(f'hive_metastore.ab_oil_spills_{initials}.morbidity_features', force_refresh=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Lab challenge
# MAGIC What conclusions can you draw from the experiment matrix? Is this experiment valid? Would it be a good candidate to use as-is, or would we want to tweak it further?

# COMMAND ----------

# DBTITLE 1,Lab Challenge Cell
#hint: Is there evidence of overfitting? Think about what we could do given how we engineered our features table to improve the model.
