# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Data loader
# MAGIC 
# MAGIC ## Use this notebook to stream input data.
# MAGIC Run this notebook and switch to DLT pipeline load data in real-time.
# MAGIC 
# MAGIC Data is incrementally loaded to BDFS under `<user_home>/anomaly_detection_sol_acc/landing/turbine-data-<increment_num>.json`
# MAGIC In this notebook you can configure the following:
# MAGIC 
# MAGIC   - **Speed**: Number of seconds between incremental batches. New data file is created after this time.
# MAGIC   - **Volume**: Amount of records in single batch.
# MAGIC   - **Write count**: Amount of incremental batches to be written.
# MAGIC   
# MAGIC Data is stored as JSON documents.
# MAGIC 
# MAGIC <b style="color:red">Don't run DLT pipeline unless this notebook already shows that is is writing data. DLT might fail with no such file or directory</b>
# MAGIC 
# MAGIC _Note: Total amount of sample data is 17.000. This is maximum amount that can be streamed._

# COMMAND ----------

#dbutils.widgets.removeAll()
dbutils.widgets.combobox('reset_all', 'False', ['True', 'False'], 'Reset All data')
dbutils.widgets.combobox('batch_wait', '3', ['3', '5', '15', '30', '45', '60'], 'Speed (secs between writes)')
dbutils.widgets.combobox('num_recs', '100', ['5000', '10000', '20000'], 'Volume (# records per writes)')
dbutils.widgets.combobox('batch_count', '100', ['100', '200', '500'], 'Write count (for how long we append data)')

# COMMAND ----------

# MAGIC %run ./_resources/00-setup