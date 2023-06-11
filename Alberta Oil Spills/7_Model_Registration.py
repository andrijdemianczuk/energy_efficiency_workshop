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
# MAGIC In this notebook we will register our best ML model for promotion through the MLOps lifecycle using MLFlow
# MAGIC
# MAGIC One of the primary challenges among data scientists and ML engineers is the absence of a central repository for models, their versions, and the means to manage them throughout their lifecycle.  
# MAGIC
# MAGIC [The MLflow Model Registry](https://docs.databricks.com/applications/mlflow/model-registry.html) addresses this challenge and enables members of the data team to:
# MAGIC <br><br>
# MAGIC * **Discover** registered models, current stage in model development, experiment runs, and associated code with a registered model
# MAGIC * **Transition** models to different stages of their lifecycle
# MAGIC * **Deploy** different versions of a registered model in different stages, offering MLOps engineers ability to deploy and conduct testing of different model versions
# MAGIC * **Test** models in an automated fashion
# MAGIC * **Document** models throughout their lifecycle
# MAGIC * **Secure** access and permission for model registrations, transitions or modifications

# COMMAND ----------

# MAGIC %md
# MAGIC ### How to Use the Model Registry
# MAGIC Typically, data scientists who use MLflow will conduct many experiments, each with a number of runs that track and log metrics and parameters. During the course of this development cycle, they will select the best run within an experiment and register its model with the registry.  Think of this as **committing** the model to the registry, much as you would commit code to a version control system.  
# MAGIC
# MAGIC The registry defines several model stages: `None`, `Staging`, `Production`, and `Archived`. Each stage has a unique meaning. For example, `Staging` is meant for model testing, while `Production` is for models that have completed the testing or review processes and have been deployed to applications. 
# MAGIC
# MAGIC Users with appropriate permissions can transition models between stages.

# COMMAND ----------

# MAGIC %run ./4_MLFlow_Helpers $reset_all_data=false $catalog="hive_metastore"

# COMMAND ----------

# DBTITLE 1,Retrieving the Best Run
#Let's get our last auto ml run. This is specific to the demo, it just gets the experiment ID of the last Auto ML run.
experiment_id = get_automl_churn_run()['experiment_id']

best_model = mlflow.search_runs(experiment_ids=[experiment_id], order_by=["metrics.val_f1_score DESC"], max_results=1, filter_string="status = 'FINISHED'")
best_model

# COMMAND ----------

# DBTITLE 1,Creating Tags & Registering Our Best Model
run_id = best_model.iloc[0]['run_id']

#add some tags that we'll reuse later to validate the model
client = mlflow.tracking.MlflowClient()
client.set_tag(run_id, key='demographic_vars', value='incident_number')
client.set_tag(run_id, key='db_table', value=f'hive_metastore.ab_oil_spills_{initials}.morbidity_features')

#Deploy our autoML run in MLFlow registry
model_details = mlflow.register_model(f"runs:/{run_id}/model", f"dbdemos_mlops_morbidity_{initials}")

# COMMAND ----------

# DBTITLE 1,Updating Our Descriptions for the Model & Version
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

# MAGIC %md-sandbox
# MAGIC #### Request Transition to Staging
# MAGIC
# MAGIC <img style="float: right" src="https://github.com/QuentinAmbard/databricks-demo/raw/main/retail/resources/images/churn_move_to_stating.gif">
# MAGIC
# MAGIC Our model is now read! Let's request a transition to Staging. 
# MAGIC
# MAGIC While this example is done using the API, we can also simply click on the Model Registry button.

# COMMAND ----------

# DBTITLE 1,Request Transition
request_transition(model_name = f"dbdemos_mlops_morbidity_{initials}", version = model_details.version, stage = "Staging")

# Leave a comment for the ML engineer who will be reviewing the tests
comment = "This was the best model from AutoML, I think we can use it as a baseline."

model_comment(model_name = f"dbdemos_mlops_morbidity_{initials}",
             version = model_details.version,
             comment = comment)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Lab challenge
# MAGIC In this example we are requesting approval transition a model to staging. What would be a good way to notify an approver?

# COMMAND ----------

# DBTITLE 1,Lab Challenge Cell
#hint: What types of automation can help with this
