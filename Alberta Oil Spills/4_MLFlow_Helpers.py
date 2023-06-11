# Databricks notebook source
# MAGIC %md
# MAGIC Table: morbidity_features
# MAGIC Target Column: is_morbid

# COMMAND ----------

# MAGIC %md
# MAGIC ## Global Setup
# MAGIC
# MAGIC The Global setup contains several supporting functions developed by the Databricks team to help programmatically manage your ML Flow experiments. Generally speaking there is no need to modify these. In time they will be added to a supporting class library.

# COMMAND ----------

initials = "ad"

# COMMAND ----------

# DBTITLE 1,Functions to Initialize the AutoML Experiment (Boilerplate)
from delta.tables import *
import pandas as pd
import logging
from pyspark.sql.functions import (
    to_date,
    col,
    regexp_extract,
    rand,
    to_timestamp,
    initcap,
    sha1,
)
from pyspark.sql.types import *
from pyspark.sql.functions import pandas_udf, PandasUDFType, input_file_name, col
import pyspark.sql.functions as F
import re
import time


# VERIFY DATABRICKS VERSION COMPATIBILITY ----------

try:
    min_required_version = dbutils.widgets.get("min_dbr_version")
except:
    min_required_version = "9.1"

version_tag = spark.conf.get("spark.databricks.clusterUsageTags.sparkVersion")
version_search = re.search("^([0-9]*\.[0-9]*)", version_tag)
assert (
    version_search
), f"The Databricks version can't be extracted from {version_tag}, shouldn't happen, please correct the regex"
current_version = float(version_search.group(1))
assert float(current_version) >= float(
    min_required_version
), f"The Databricks version of the cluster must be >= {min_required_version}. Current version detected: {current_version}"
assert (
    "ml" in version_tag.lower()
), f"The Databricks ML runtime must be used. Current version detected doesn't contain 'ml': {version_tag} "


# python Imports for ML...
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import mlflow
import mlflow.sklearn
from mlflow.tracking.client import MlflowClient
from hyperopt import fmin, hp, tpe, STATUS_OK, Trials
from hyperopt.pyll.base import scope
from hyperopt import SparkTrials
from sklearn.model_selection import GroupKFold
from pyspark.sql.functions import pandas_udf, PandasUDFType
import os
import pandas as pd
from hyperopt import space_eval
import numpy as np
from time import sleep


from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix

# force the experiment to the field demos one. Required to launch as a batch
def init_experiment_for_batch(demo_name, experiment_name):
    # You can programatically get a PAT token with the following
    pat_token = (
        dbutils.notebook.entry_point.getDbutils()
        .notebook()
        .getContext()
        .apiToken()
        .get()
    )
    url = (
        dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
    )
    # current_user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
    import requests

    xp_root_path = f"/dbdemos/experiments/{demo_name}"
    requests.post(
        f"{url}/api/2.0/workspace/mkdirs",
        headers={"Accept": "application/json", "Authorization": f"Bearer {pat_token}"},
        json={"path": xp_root_path},
    )
    xp = f"{xp_root_path}/{experiment_name}"
    print(f"Using common experiment under {xp}")
    mlflow.set_experiment(xp)
    return mlflow.get_experiment_by_name(xp)

def get_cloud_name():
  return spark.conf.get("spark.databricks.clusterUsageTags.cloudProvider").lower()

# COMMAND ----------

# DBTITLE 1,Functions to Manage AutoML Experiment (Boilerplate)
from pyspark.sql.functions import col
import mlflow

import databricks
from datetime import datetime


def get_automl_run(name):
    # get the most recent automl run
    df = (
        spark.table("hive_metastore.dbdemos_metadata.automl_experiment")
        .filter(col("name") == name)
        .orderBy(col("date").desc())
        .limit(1)
    )
    return df.collect()


# Get the automl run information from the hive_metastore.dbdemos_metadata.automl_experiment table.
# If it's not available in the metadata table, start a new run with the given parameters
def get_automl_run_or_start(
    name, model_name, dataset, target_col, timeout_minutes, move_to_production=False
):
    spark.sql("create database if not exists hive_metastore.dbdemos_metadata")
    spark.sql(
        "create table if not exists hive_metastore.dbdemos_metadata.automl_experiment (name string, date string)"
    )
    result = get_automl_run(name)
    if len(result) == 0:
        print(
            "No run available, start a new Auto ML run, this will take a few minutes..."
        )
        start_automl_run(
            name, model_name, dataset, target_col, timeout_minutes, move_to_production
        )
        return (False, get_automl_run(name))
    return (True, result[0])


# Start a new auto ml classification task and save it as metadata.
def start_automl_run(
    name, model_name, dataset, target_col, timeout_minutes=5, move_to_production=False
):
    from databricks import automl

    automl_run = databricks.automl.classify(
        dataset=dataset, target_col=target_col, timeout_minutes=timeout_minutes
    )
    experiment_id = automl_run.experiment.experiment_id
    path = automl_run.experiment.name
    data_run_id = (
        mlflow.search_runs(
            experiment_ids=[automl_run.experiment.experiment_id],
            filter_string="tags.mlflow.source.name='Notebook: DataExploration'",
        )
        .iloc[0]
        .run_id
    )
    exploration_notebook_id = automl_run.experiment.tags[
        "_databricks_automl.exploration_notebook_id"
    ]
    best_trial_notebook_id = automl_run.experiment.tags[
        "_databricks_automl.best_trial_notebook_id"
    ]

    cols = [
        "name",
        "date",
        "experiment_id",
        "experiment_path",
        "data_run_id",
        "best_trial_run_id",
        "exploration_notebook_id",
        "best_trial_notebook_id",
    ]
    spark.createDataFrame(
        data=[
            (
                name,
                datetime.today().isoformat(),
                experiment_id,
                path,
                data_run_id,
                automl_run.best_trial.mlflow_run_id,
                exploration_notebook_id,
                best_trial_notebook_id,
            )
        ],
        schema=cols,
    ).write.mode("append").option("mergeSchema", "true").saveAsTable(
        "hive_metastore.dbdemos_metadata.automl_experiment"
    )
    # Create & save the first model version in the MLFlow repo (required to setup hooks etc)
    model_registered = mlflow.register_model(
        f"runs:/{automl_run.best_trial.mlflow_run_id}/model", model_name
    )
    set_experiment_permission(path)
    if move_to_production:
        client = mlflow.tracking.MlflowClient()
        print(
            "registering model version "
            + model_registered.version
            + " as production model"
        )
        client.transition_model_version_stage(
            name=model_name,
            version=model_registered.version,
            stage="Production",
            archive_existing_versions=True,
        )
    return get_automl_run(name)


# Generate nice link for the given auto ml run
def display_automl_link(
    name, model_name, dataset, target_col, timeout_minutes=5, move_to_production=False
):
    from_cache, r = get_automl_run_or_start(
        name, model_name, dataset, target_col, timeout_minutes, move_to_production
    )
    if from_cache:
        html = f"""For exploratory data analysis, open the <a href="/#notebook/{r["exploration_notebook_id"]}">data exploration notebook</a><br/><br/>"""
        html += f"""To view the best performing model, open the <a href="/#notebook/{r["best_trial_notebook_id"]}">best trial notebook</a><br/><br/>"""
        html += f"""To view details about all trials, navigate to the <a href="/#mlflow/experiments/{r["experiment_id"]}/s?orderByKey=metrics.%60val_f1_score%60&orderByAsc=false">MLflow experiment</>"""
        displayHTML(html)


def reset_automl_run(model_name):
    if spark._jsparkSession.catalog().tableExists(
        "hive_metastore.dbdemos_metadata.automl_experiment"
    ):
        spark.sql(
            f"delete from hive_metastore.dbdemos_metadata.automl_experiment where name='{model_name}'"
        )


# Once the automl experiment is created, we assign CAN MANAGE to all users as it's shared in the workspace
def set_experiment_permission(experiment_path):
    url = (
        dbutils.notebook.entry_point.getDbutils()
        .notebook()
        .getContext()
        .extraContext()
        .apply("api_url")
    )
    import requests

    pat_token = (
        dbutils.notebook.entry_point.getDbutils()
        .notebook()
        .getContext()
        .apiToken()
        .get()
    )
    headers = {
        "Authorization": "Bearer " + pat_token,
        "Content-type": "application/json",
    }
    status = requests.get(
        url + "/api/2.0/workspace/get-status",
        params={"path": experiment_path},
        headers=headers,
    ).json()
    # Set can manage to all users to the experiment we created as it's shared among all
    params = {
        "access_control_list": [
            {"group_name": "users", "permission_level": "CAN_MANAGE"}
        ]
    }
    permissions = requests.patch(
        f"{url}/api/2.0/permissions/experiments/{status['object_id']}",
        json=params,
        headers=headers,
    )
    if permissions.status_code != 200:
        print("ERROR: couldn't set permission to all users to the autoML experiment")

    # try to find the experiment id
    result = re.search(
        r"_([a-f0-9]{8}_[a-f0-9]{4}_[a-f0-9]{4}_[a-f0-9]{4}_[a-f0-9]{12})_",
        experiment_path,
    )
    if result is not None and len(result.groups()) > 0:
        ex_id = result.group(0)
    else:
        print(experiment_path)
        ex_id = experiment_path[experiment_path.rfind("/") + 1 :]

    path = experiment_path
    path = path[: path.rfind("/")] + "/"
    # List to get the folder with the notebooks from the experiment
    folders = requests.get(
        url + "/api/2.0/workspace/list", params={"path": path}, headers=headers
    ).json()
    for f in folders["objects"]:
        if f["object_type"] == "DIRECTORY" and ex_id in f["path"]:
            # Set the permission of the experiment notebooks to all
            permissions = requests.patch(
                f"{url}/api/2.0/permissions/directories/{f['object_id']}",
                json=params,
                headers=headers,
            )
            if permissions.status_code != 200:
                print(
                    "ERROR: couldn't set permission to all users to the autoML experiment notebooks"
                )

# COMMAND ----------

# DBTITLE 1,DBUtils Supporting Functions (Boilerplate)
import warnings

def test_not_empty_folder(folder):
    try:
        return len(dbutils.fs.ls(folder)) > 0
    except:
        return False


# Return true if the folder is empty or does not exists
def is_folder_empty(folder):
    try:
        return len(dbutils.fs.ls(folder)) == 0
    except:
        return True

with warnings.catch_warnings():
    warnings.simplefilter('ignore', SyntaxWarning)
    warnings.simplefilter('ignore', DeprecationWarning)
    warnings.simplefilter('ignore', UserWarning)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

# DBTITLE 1,Import and Validate the ML Flow Libraries and Dependencies
import mlflow
if "evaluate" not in dir(mlflow):
    raise Exception("ERROR - YOU NEED MLFLOW 2.0 for this demo. Select DBRML 12+")
    
from databricks.feature_store import FeatureStoreClient
from mlflow import MlflowClient
from mlflow.utils.rest_utils import http_request
import json

# COMMAND ----------

# DBTITLE 1,Entry Point Functions for AutoML
def display_automl_churn_link(
    table_name,
    force_refresh=False,
    xp_name=f"morbidity_features_{initials}",
    model_name=f"dbdemos_mlops_morbidity_{initials}",
    target_col="is_morbid",
):
    timeout = 5
    if force_refresh:
        reset_automl_run(xp_name)
    display_automl_link(
        xp_name, model_name, spark.table(table_name), target_col, timeout
    )


def get_automl_churn_run(
    table_name=f"hive_metastore.ab_oil_spills_{initials}.morbidity_features",
    force_refresh=False,
    xp_name=f"morbidity_features_{initials}",
    model_name=f"dbdemos_mlops_morbidity_{initials}",
    target_col="is_morbid",
):
    timeout = 5
    if force_refresh:
        reset_automl_run(xp_name)
    from_cache, r = get_automl_run_or_start(
        xp_name, model_name, spark.table(table_name), target_col, timeout
    )
    return r

# COMMAND ----------

# MAGIC %md
# MAGIC ## API Helpers

# COMMAND ----------

# DBTITLE 1,Define the Job Helper Functions (Boilerplate)
# Helper to get the MLOps Databricks job or create it if it doesn't exists
def find_job(name, offset=0, limit=25):
    r = http_request(
        host_creds=host_creds,
        endpoint="/api/2.1/jobs/list",
        method="GET",
        params={
            "limit": limit,
            "offset": offset,
            "name": urllib.parse.quote_plus(name),
        },
    ).json()
    if "jobs" in r:
        for job in r["jobs"]:
            if job["settings"]["name"] == name:
                return job
        if r["has_more"]:
            return find_job(name, offset + limit, limit)
    return None


def get_churn_staging_job_id():
    job = find_job(f"demos_morbidity_model_staging_validation_{initials}")
    if job is not None:
        return job["job_id"]
    else:
        # the job doesn't exist, we dynamically create it.
        # Note: requires DBR 10 ML to use automl model
        notebook_path = (
            dbutils.entry_point.getDbutils()
            .notebook()
            .getContext()
            .notebookPath()
            .get()
        )
        base_path = "/".join(notebook_path.split("/")[:-1])
        cloud_name = get_cloud_name()
        if cloud_name == "aws":
            node_type = "i3.xlarge"
        elif cloud_name == "azure":
            node_type = "Standard_DS3_v2"
        elif cloud_name == "gcp":
            node_type = "n1-standard-4"
        else:
            raise Exception(f"Cloud '{cloud_name}' isn't supported!")
        job_settings = {
            "email_notifications": {},
            "name": f"demos_morbidity_model_staging_validation_{initials}",
            "max_concurrent_runs": 1,
            "tasks": [
                {
                    "new_cluster": {
                        "spark_version": "12.2.x-cpu-ml-scala2.12",
                        "spark_conf": {
                            "spark.databricks.cluster.profile": "singleNode",
                            "spark.master": "local[*, 4]",
                        },
                        "num_workers": 0,
                        "node_type_id": node_type,
                        "driver_node_type_id": node_type,
                        "custom_tags": {"ResourceClass": "SingleNode"},
                        "spark_env_vars": {
                            "PYSPARK_PYTHON": "/databricks/python3/bin/python3"
                        },
                        "enable_elastic_disk": True,
                    },
                    "notebook_task": {
                        "notebook_path": f"{base_path}/05_job_staging_validation"
                    },
                    "email_notifications": {},
                    "task_key": "test-model",
                }
            ],
        }
        print("Job doesn't exists, creating it...")
        r = http_request(
            host_creds=host_creds,
            endpoint="/api/2.1/jobs/create",
            method="POST",
            json=job_settings,
        ).json()
        return r["job_id"]

# COMMAND ----------

# DBTITLE 1,Define the Job Hooks to Support Experiment Management (Boilerplate)
# Manage webhooks
try:
    from databricks_registry_webhooks import (
        RegistryWebhooksClient,
        JobSpec,
        HttpUrlSpec,
    )

    def create_job_webhook(model_name, job_id):
        return RegistryWebhooksClient().create_webhook(
            model_name=model_name,
            events=["TRANSITION_REQUEST_CREATED"],
            job_spec=JobSpec(job_id=job_id, access_token=token),
            description="Trigger the ops_validation job when a model is requested to move to staging.",
            status="ACTIVE",
        )

    def create_notification_webhook(model_name, slack_url):
        from databricks_registry_webhooks import (
            RegistryWebhooksClient,
            JobSpec,
            HttpUrlSpec,
        )

        return RegistryWebhooksClient().create_webhook(
            model_name=model_name,
            events=["TRANSITION_REQUEST_CREATED"],
            description="Notify the MLOps team that a model is requested to move to staging.",
            status="ACTIVE",
            http_url_spec=HttpUrlSpec(url=slack_url),
        )

    # List
    def list_webhooks(model_name):
        from databricks_registry_webhooks import RegistryWebhooksClient

        return RegistryWebhooksClient().list_webhooks(model_name=model_name)

    # Delete
    def delete_webhooks(webhook_id):
        from databricks_registry_webhooks import RegistryWebhooksClient

        return RegistryWebhooksClient().delete_webhook(id=webhook_id)

except:

    def raise_exception():
        print(
            "You need to install databricks-registry-webhooks library to easily perform this operation (you could also use the rest API directly)."
        )
        print("Please run: %pip install databricks-registry-webhooks ")
        raise RuntimeError(
            "function not available without databricks-registry-webhooks."
        )

    def create_job_webhook(model_name, job_id):
        raise_exception()

    def create_notification_webhook(model_name, slack_url):
        raise_exception()

    def list_webhooks(model_name):
        raise_exception()

    def delete_webhooks(webhook_id):
        raise_exception()


def reset_webhooks(model_name):
    whs = list_webhooks(model_name)
    for wh in whs:
        delete_webhooks(wh.id)

# COMMAND ----------

# DBTITLE 1,MLOps Helpers (Boilerplate)
client = mlflow.tracking.client.MlflowClient()

host_creds = client._tracking_client.store.get_host_creds()
host = host_creds.host
token = host_creds.token


def mlflow_call_endpoint(endpoint, method, body="{}"):
    if method == "GET":
        response = http_request(
            host_creds=host_creds,
            endpoint="/api/2.0/mlflow/{}".format(endpoint),
            method=method,
            params=json.loads(body),
        )
    else:
        response = http_request(
            host_creds=host_creds,
            endpoint="/api/2.0/mlflow/{}".format(endpoint),
            method=method,
            json=json.loads(body),
        )
    return response.json()


# Request transition to staging
def request_transition(model_name, version, stage):

    staging_request = {
        "name": model_name,
        "version": version,
        "stage": stage,
        "archive_existing_versions": "true",
    }
    response = mlflow_call_endpoint(
        "transition-requests/create", "POST", json.dumps(staging_request)
    )
    return response


# Comment on model
def model_comment(model_name, version, comment):

    comment_body = {"name": model_name, "version": version, "comment": comment}
    response = mlflow_call_endpoint("comments/create", "POST", json.dumps(comment_body))
    return response


# Accept or reject transition request
def accept_transition(model_name, version, stage, comment):
    approve_request_body = {
        "name": model_details.name,
        "version": model_details.version,
        "stage": stage,
        "archive_existing_versions": "true",
        "comment": comment,
    }

    mlflow_call_endpoint(
        "transition-requests/approve", "POST", json.dumps(approve_request_body)
    )


def reject_transition(model_name, version, stage, comment):

    reject_request_body = {
        "name": model_details.name,
        "version": model_details.version,
        "stage": stage,
        "comment": comment,
    }

    mlflow_call_endpoint(
        "transition-requests/reject", "POST", json.dumps(reject_request_body)
    )
