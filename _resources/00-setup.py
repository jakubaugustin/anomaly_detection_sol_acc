# Databricks notebook source
# MAGIC %pip install kaggle

# COMMAND ----------

# DBTITLE 1,Initiate widgets
# dbutils.widgets.removeAll()

dbutils.widgets.combobox('reset_all', 'False', ['True', 'False'], 'Reset All data')
dbutils.widgets.combobox('batch_wait', '3', ['3', '5', '15', '30', '45', '60'], 'Speed (secs between writes)')
dbutils.widgets.combobox('num_recs', '100', ['5000', '10000', '20000'], 'Volume (# records per writes)')
dbutils.widgets.combobox('batch_count', '100', ['100', '200', '500'], 'Write count (for how long we append data)')

pass

# COMMAND ----------

# DBTITLE 1,Package imports - used across the solution accelerator
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import re
import time
from datetime import datetime

# COMMAND ----------

# DBTITLE 1,Check minimum supported version
# Set minimal supported version
min_required_version = "9.1"
ml_required = False


version_tag = spark.conf.get("spark.databricks.clusterUsageTags.sparkVersion")
version_search = re.search('^([0-9]*\.[0-9]*)', version_tag)
assert version_search, f"The Databricks version can't be extracted from {version_tag}, shouldn't happen, please correct the regex"
current_version = float(version_search.group(1))
assert float(current_version) >= float(min_required_version), f'The Databricks version of the cluster must be >= {min_required_version}. Current version detected: {current_version}'
assert not ml_required or "ml" in version_tag.lower(), f"The Databricks ML runtime must be used. Current version detected doesn't contain 'ml': {version_tag} "

print(f'DBR version {version_tag} check: ' + u'\u2713')

# COMMAND ----------

# DBTITLE 1,Initiate helper functions
def get_cloud_name():
  ## Get cloud name
  return spark.conf.get("spark.databricks.clusterUsageTags.cloudProvider").lower()

def get_current_user(with_at = True):
  ## Get current user with or without '@'
  current_user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
  if current_user.rfind('@') > 0:
    current_user_no_at = current_user[:current_user.rfind('@')]
  else:
    current_user_no_at = current_user
  current_user_no_at = re.sub(r'\W+', '_', current_user_no_at)
  
  if with_at:
    out = current_user
  else:
    out = current_user_no_at
    
  return out


def display_slide(slide_id, slide_number):
  ## Display google slide as HTML in notebook
  displayHTML(f'''
  <div style="width:1150px; margin:auto">
  <iframe
    src="https://docs.google.com/presentation/d/{slide_id}/embed?slide={slide_number}"
    frameborder="0"
    width="1150"
    height="683"
  ></iframe></div>
  ''')
  

def stop_all_streams():
  ## Function to stop all streaming queries 
  stream_count = len(spark.streams.active)
  if stream_count > 0:
    print(f"Stopping {stream_count} streams")
    for s in spark.streams.active:
        try:
            s.stop()
        except:
            pass
    print("All stream stopped.")
    
    
def wait_for_all_stream(start = ""):
  ## Wait for all streams to stop
  import time
  def get_active_streams(start):
    return [s for s in spark.streams.active if len(start) == 0 or (s.name is not None and s.name.startswith(start))]
  actives = get_active_streams(start)
  if len(actives) > 0:
    print(f"{len(actives)} streams still active, waiting... ({[s.name for s in actives]})")
  while len(actives) > 0:
    spark.streams.awaitAnyTermination()
    time.sleep(1)
    actives = get_active_streams(start)
  print("All streams completed.")
  
def test_not_empty_folder(folder):
  try:
    return len(dbutils.fs.ls(folder)) > 0
  except:
    return False
  
print(f'Helper functions initiated: ' + u'\u2713')

# COMMAND ----------

# DBTITLE 1,Helpers for AutoML runs
from pyspark.sql.functions import col
from databricks.feature_store import FeatureStoreClient
import mlflow

import databricks
from databricks import automl
from datetime import datetime

def get_automl_run(name):
  #get the most recent automl run
  df = spark.table("field_demos_metadata.automl_experiment").filter(col("name") == name).orderBy(col("date").desc()).limit(1)
  return df.collect()

#Get the automl run information from the field_demos_metadata.automl_experiment table. 
#If it's not available in the metadata table, start a new run with the given parameters
def get_automl_run_or_start(name, model_name, dataset, target_col, timeout_minutes, move_to_production = False):
  spark.sql("create database if not exists field_demos_metadata")
  spark.sql("create table if not exists field_demos_metadata.automl_experiment (name string, date string)")
  result = get_automl_run(name)
  if len(result) == 0:
    print("No run available, start a new Auto ML run, this will take a few minutes...")
    start_automl_run(name, model_name, dataset, target_col, timeout_minutes, move_to_production)
    result = get_automl_run(name)
  return result[0]


#Start a new auto ml classification task and save it as metadata.
def start_automl_run(name, model_name, dataset, target_col, timeout_minutes = 5, move_to_production = False):
  automl_run = databricks.automl.classify(
    dataset = dataset,
    target_col = target_col,
    timeout_minutes = timeout_minutes
  )
  experiment_id = automl_run.experiment.experiment_id
  path = automl_run.experiment.name
  data_run_id = mlflow.search_runs(experiment_ids=[automl_run.experiment.experiment_id], filter_string = "tags.mlflow.source.name='Notebook: DataExploration'").iloc[0].run_id
  exploration_notebook_id = automl_run.experiment.tags["_databricks_automl.exploration_notebook_id"]
  best_trial_notebook_id = automl_run.experiment.tags["_databricks_automl.best_trial_notebook_id"]

  cols = ["name", "date", "experiment_id", "experiment_path", "data_run_id", "best_trial_run_id", "exploration_notebook_id", "best_trial_notebook_id"]
  spark.createDataFrame(data=[(name, datetime.today().isoformat(), experiment_id, path, data_run_id, automl_run.best_trial.mlflow_run_id, exploration_notebook_id, best_trial_notebook_id)], schema = cols).write.mode("append").option("mergeSchema", "true").saveAsTable("field_demos_metadata.automl_experiment")
  #Create & save the first model version in the MLFlow repo (required to setup hooks etc)
  model_registered = mlflow.register_model(f"runs:/{automl_run.best_trial.mlflow_run_id}/model", model_name)
  if move_to_production:
    client = mlflow.tracking.MlflowClient()
    print("registering model version "+model_registered.version+" as production model")
    client.transition_model_version_stage(name = model_name, version = model_registered.version, stage = "Production", archive_existing_versions=True)
  return get_automl_run(name)

#Generate nice link for the given auto ml run
def display_automl_link(name, model_name, dataset, target_col, timeout_minutes = 5, move_to_production = False):
  r = get_automl_run_or_start(name, model_name, dataset, target_col, timeout_minutes, move_to_production)
  html = f"""For exploratory data analysis, open the <a href="/#notebook/{r["exploration_notebook_id"]}">data exploration notebook</a><br/><br/>"""
  html += f"""To view the best performing model, open the <a href="/#notebook/{r["best_trial_notebook_id"]}">best trial notebook</a><br/><br/>"""
  html += f"""To view details about all trials, navigate to the <a href="/#mlflow/experiments/{r["experiment_id"]}/s?orderByKey=metrics.%60val_f1_score%60&orderByAsc=false">MLflow experiment</>"""
  displayHTML(html)

def reset_automl_run(model_name):
  spark.sql(f"delete from field_demos_metadata.automl_experiment where name='{model_name}'")

# COMMAND ----------

# DBTITLE 1,Mount Filed demos
mount_name = "field-demos"

try:
  dbutils.fs.ls("/mnt/%s" % mount_name)
except:
  workspace_id = dbutils.entry_point.getDbutils().notebook().getContext().workspaceId().get()
  url = dbutils.entry_point.getDbutils().notebook().getContext().apiUrl().get()
  if workspace_id == '8194341531897276':
    print("CSE2 bucket isn't mounted, mount the demo data under %s" % mount_name)
    dbutils.fs.mount(f"s3a://databricks-field-demos/" , f"/mnt/{mount_name}")
  elif "azure" in url:
    print("ADLS2 isn't mounted, mount the demo data under %s" % mount_name)
    configs = {"fs.azure.account.auth.type": "OAuth",
              "fs.azure.account.oauth.provider.type": "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider",
              "fs.azure.account.oauth2.client.id": dbutils.secrets.get(scope = "common-sp", key = "common-sa-sp-client-id"),
              "fs.azure.account.oauth2.client.secret": dbutils.secrets.get(scope = "common-sp", key = "common-sa-sp-client-secret"),
              "fs.azure.account.oauth2.client.endpoint": "https://login.microsoftonline.com/9f37a392-f0ae-4280-9796-f1864a10effc/oauth2/token"}

    dbutils.fs.mount(
      source = "abfss://field-demos@fielddemosdatasets.dfs.core.windows.net/field-demos",
      mount_point = "/mnt/"+mount_name,
      extra_configs = configs)
  else:
    aws_bucket_name = ""
    print("bucket isn't mounted, mount the demo bucket under %s" % mount_name)
    dbutils.fs.mount(f"s3a://databricks-datasets-private/field-demos" , f"/mnt/{mount_name}")
    
print(f'Source data mounted: ' + u'\u2713')

# COMMAND ----------

# MAGIC %sh
# MAGIC export KAGGLE_USERNAME=<kaggle_user_here>
# MAGIC export KAGGLE_KEY=<kaggle_secret_here>
# MAGIC 
# MAGIC #kaggle datasets download --force  -d berkerisen/wind-turbine-scada-dataset
# MAGIC kaggle datasets download --force -d psycon/wind-turbine-energy-kw-generation-data
# MAGIC 
# MAGIC unzip -f wind-turbine-energy-kw-generation-data.zip -d ./kaggle_dataset

# COMMAND ----------

# MAGIC %sh
# MAGIC 
# MAGIC ./kaggle_dataset
# MAGIC unzip -o wind-turbine-energy-kw-generation-data.zip -d ./kaggle_dataset
# MAGIC 
# MAGIC ls /databricks/driver/

# COMMAND ----------

# DBTITLE 1,Ensure data is ready in source directory
user_name_at = get_current_user(True)
user_name_no_at = get_current_user(False)

source_data_location = f"/user/{user_name_at}/anomaly_detection_sol_acc"
source_data_location_turbine = f"{source_data_location}/incomming-data"
# source_data_location_status = f"{source_data_location}/status"
db_name = f"{user_name_no_at}_anomaly_detection_sol_acc"

# Clean DB and data directory when clean_all set to True
if dbutils.widgets.get("reset_all") == 'True':
  spark.sql(f"DROP DATABASE IF EXISTS {db_name} CASCADE")
  dbutils.fs.rm(source_data_location, True)

# Load data from /mnt location
# TODO: handle state with missing data in /mnt
data_available = test_not_empty_folder(source_data_location_turbine) 
if not data_available:
  dbutils.fs.cp("file:/databricks/driver/kaggle_dataset/features.csv", source_data_location_turbine, True)
  # dbutils.fs.cp("/mnt/field-demos/manufacturing/iot_turbine/status", source_data_location_status, True)
  
spark.sql(f"""create database if not exists {db_name} LOCATION '{source_data_location}/tables' """)
spark.sql(f"""USE {db_name}""")

print(f'Data loaded and DB created: ' + u'\u2713')
print(f'  - Data path: {source_data_location_turbine}')
print(f'  - DB name: {db_name}')

# COMMAND ----------

# MAGIC %sh
# MAGIC 
# MAGIC rm -f ./kaggle_dataset/*
# MAGIC rmdir ./kaggle_dataset

# COMMAND ----------

# DBTITLE 1,Load source to DF
from pyspark.sql.window import Window
import pyspark.sql.functions as F

turbine_data_df = spark.read.option("header", "True").csv(source_data_location_turbine).na.drop()
cols = turbine_data_df.columns
#cast cols to numericals and timestamp
to_select = [F.to_timestamp(F.col("Timestamp")).alias("timestamp")]
for c in cols:
  if c != "Timestamp":
    to_select.append(F.col(c).cast("double").alias(c.replace('(', '').replace(')', '').replace(' ', '_')))
    
initial_timestamp = 1640991600    
turbine_data_df = turbine_data_df.withColumn("rn", F.row_number().over(Window.orderBy(F.col("timestamp"))))
turbine_data_df = turbine_data_df.withColumn("ts_new", F.from_unixtime(F.lit(initial_timestamp) + F.col("rn") * 10))

turbine_data_df = turbine_data_df.drop("timestamp")
turbine_data_df =  turbine_data_df.withColumnRenamed("ts_new", "timestamp")
  
turbine_data_df = turbine_data_df.select(*to_select)
r_count = turbine_data_df.count()
print(f"Data row count: {r_count}")

batch_wait = int(dbutils.widgets.get("batch_wait"))
num_recs = int(dbutils.widgets.get("num_recs"))
batch_count = int(dbutils.widgets.get("batch_count"))
row_count = num_recs * batch_count

if row_count > r_count:
  print(f'Expected number of rows to be written ({row_count}) is higher that total rows in source dataset ({r_count}).')
  
# Limit to expected number of rows and sort by date
# Compute row group - records in same row group will be exported in same file
turbine_data_df = turbine_data_df.limit(row_count).withColumn('row_group', F.ceil(F.row_number().over(Window.partitionBy().orderBy(turbine_data_df['timestamp'])) / F.lit(num_recs))).sort(turbine_data_df['timestamp'])

# COMMAND ----------

turbine_data_df.printSchema()

# COMMAND ----------

# DBTITLE 1,Stream batch to landing directory
print(f'Data generator statred: ' + u'\u2713')
landing_data_location = f"{source_data_location}/landing"
landing_data_location_python = f"/dbfs{landing_data_location}"
print(f'  - writing data to: {landing_data_location}')

# Clear landing data and recreate the directory
# dbutils.fs.rm(landing_data_location, True)
dbutils.fs.mkdirs(landing_data_location)

# Put data to pandas for eaier manipulation 
data_pd = turbine_data_df.toPandas()
data_pd['timestamp'] = data_pd['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

for i in range(batch_count):
  #to_write = turbine_data_df.filter(turbine_data_df['row_group'] == F.lit(i + 1)).drop('row_group').repartition(1)
  #to_write.write.format('parquet').mode('overwrite').save(landing_data_location)
  to_write = data_pd[data_pd['row_group'] == i + 1]
  file_name = f'{landing_data_location_python}/turbine-data-{i+1}.json'
  to_write.to_json(file_name, orient='records')
  current_time = datetime.now().strftime("%H:%M:%S")
  min_ts_value = to_write['timestamp'].min()
  max_ts_value = to_write['timestamp'].max()
  print(f'{current_time}: Now writing batch {i+1}/{batch_count} of {len(to_write.index)} records.')
  print(f'  - timestamp values written: {min_ts_value} to {max_ts_value}')
  print(f'  - sleeping for {batch_wait} seconds')
  time.sleep(batch_wait)
  
print('Writing data finished')