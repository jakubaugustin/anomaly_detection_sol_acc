# Databricks notebook source
# MAGIC %run ./_resources/00-common-setup

# COMMAND ----------

# DBTITLE 1,Settings - initial value settings
# Length of the sliding window
time_span = "1 days"

# Period when the sliding window will re-compute
run_check_new_data = "60 minutes"

# Table to use for anoaly detection
streaming_table_name = "hive_metastore.jakub_augustin_anomaly_detection_sol_acc.turbine_silver_dlt"

# Table to use for output persistence
output_table_name = "hive_metastore.jakub_augustin_anomaly_detection_sol_acc.turbine_anomaly"

# Checkpoin location
checkpoint_location = "/user/jakub.augustin@databricks.com/anomaly_detection_sol_acc/write_checkpoint"

# List of columns to detect anomalies at
anomaly_detection_columns = ['Torque', 'Wind_Deviation_1_seconds', 'Tower_Acceleration_Normal']

# Sliding window definition
window = F.window("TIMESTAMP", time_span, run_check_new_data)

# Lookback time to identify anomalies in minutes
# set to 1 hours
sliding_window_time = 1 * 60 

# Timestamp column name
ts_column_name = "TIMESTAMP"

# COMMAND ----------

# DBTITLE 1,1: Twitter anomaly detection method
def twitter_anomaly(df_input
  , input_col_name
  , output_col_name='anomaly_twitter'
  , timestamp_col_name='TS'
  , weight=1.0):
  """
  Twitter anomaly detection mehod.
    :param df_input: Pandas dataframe with input data
    :param input_col_name: Name of column where anomaly will be detected
    :param output_col_name: Name of column indicating anomaly
    :parm timestamp_col_name: Name of column that contains timestamp value
    :param weight: Method weight when voting between multiple methods takes place
  """

  alpha = 0.025
  obs_per_period = 24
  max_anomalies = 0.02
  
  
  # Make a copy of input dataframe
  df_output = df_input.copy()
  
  window_length = len(df_output)

  
  df_output.sort_values(by=timestamp_col_name, inplace=True)
  time_max = df_output[timestamp_col_name].max() - timedelta(minutes=sliding_window_time)
  if window_length < 2 * obs_per_period:
    df_output[output_col_name] = 42
    print('Skipping')
    return df_output.loc[df_output[timestamp_col_name] >= time_max]
  
  data = df_output[input_col_name].copy()
  data.index = df_output.index.values
  decomposed = sm.tsa.seasonal_decompose(data, period=obs_per_period, two_sided=False)
  #smoothed = data - decomposed.resid.fillna(0)
  # data = data - decomposed.seasonal - data.mean()
  data = decomposed.resid.fillna(0)
  # data = data-data.mean()
  max_outliers = int(np.trunc(data.size * max_anomalies))
  R_idx = pd.Series(dtype=float)

  for i in range(1, max_outliers + 1):

    if not data.mad():
        break

    ares = abs(data.median() - data)

    ares = ares / data.mad()
    tmp_anom_index = ares[ares.values == ares.max()].index
    cand = pd.Series(data.loc[tmp_anom_index], index=tmp_anom_index)
    data.drop(tmp_anom_index, inplace=True)

    # Compute critical value.
    p = 1 - alpha / (2 * (window_length - i + 1))
    t = sp.stats.t.ppf(p, window_length - i - 1)
    lam = t * (window_length - i) / np.sqrt((window_length - i - 1 + t ** 2) * (window_length - i + 1))
    
    if ares.max() > lam:
      R_idx = R_idx.append(cand)
  R_idx.name = output_col_name
  df_output = df_input.merge(pd.DataFrame(R_idx), how='left', left_index=True, right_index=True).fillna(0)
  #logger.info('Processed %s, a Time-Series %d long in %0.3f secs.' % (X['context'].values[0], df_input.shape[0], time.time() - th))
  
  df_output[output_col_name] = df_output[output_col_name] > 0
  df_output[output_col_name] = df_output[output_col_name].astype(int)
  
  # return df_output.loc[df_output[timestamp_col_name] >= time_max]
  return df_output

# COMMAND ----------

# DBTITLE 1,2: Rolling MAD anomaly detection method
def rolling_mad_anomaly(
  df_input
  , input_col_name
  , output_col_name='anomaly_rolling_mad'
  , timestamp_col_name='TS'
  , weight=1.0):
  """
  Rolling median absolute deviation anomaly detection mehod.
    :param df_input: Pandas dataframe with input data
    :param input_col_name: Name of column where anomaly will be detected
    :param output_col_name: Name of column indicating anomaly
    :parm timestamp_col_name: Name of column that contains timestamp value
    :param weight: Method weight when voting between multiple methods takes place
  """
  
  # Make a copy of input dataframe
  df_output = df_input.copy()
  
  # Set initial constants
  # TODO: Describe std_th value
  std_thr = 3
  # TODO: Describe obs_per_period value
  obs_per_period = 144
  window_length = len(df_input)
  
  # Zero out anomaly col
  
  time_max = df_output[timestamp_col_name].max() - timedelta(minutes=sliding_window_time)
  df_output[output_col_name] = 0

  if window_length < 2 * obs_per_period:
    df_output[output_col_name] = 42
    return df_output[df_output[timestamp_col_name] >= time_max]    
  
  df_output.sort_values(by=timestamp_col_name, inplace=True)

  std = df_output.loc[df_output[timestamp_col_name] < time_max][input_col_name].mad()
  median = df_output.loc[df_output[timestamp_col_name] < time_max][input_col_name].median()
  df_output.loc[(df_output[timestamp_col_name] >= time_max) & (np.abs(df_output[input_col_name] - median) > std_thr * std), output_col_name] = 1.0
  print(df_output)
#logger.info('Processed %s, a Time-Series %d long in %0.3f secs.' % (X['context'].values[0], df_input.shape[0], time.time() - th))

  # return df_output[df_output[timestamp_col_name] >= time_max]
  return df_output

# COMMAND ----------

# DBTITLE 1,3: RPCA anomaly detection method
def rpca_anomaly(df_input
  , input_col_name
  , output_col_name='anomaly_rpca'
  , timestamp_col_name='TS'
  , weight=1.0):
  """
  RPCA anomaly detection mehod.
    :param df_input: Pandas dataframe with input data
    :param input_col_name: Name of column where anomaly will be detected
    :param output_col_name: Name of column indicating anomaly
    :parm timestamp_col_name: Name of column that contains timestamp value
    :param weight: Method weight when voting between multiple methods takes place
  """
  
  # Make a copy of input dataframe
  df_output = df_input.copy()

  lags = 6
  window_length = len(df_output) // lags

  
  df_output.sort_values(by=timestamp_col_name, inplace=True)
  time_max = df_output[timestamp_col_name].max() - timedelta(minutes=sliding_window_time)
  if window_length < 36 :
    print('Skipping')
    df_output[output_col_name] = 42
    return     df_output.loc[df_output[timestamp_col_name] >= time_max]

  def vector_to_matrix(x, rows, cols):
      return x.reshape(rows, cols, order='F')

  def soft_threshold(x, penalty):
      res = []
      for k in range(len(x)):
          if x[k] > 0:
              res += [max(abs(x[k]) - penalty, 0)]
          elif x[k] < 0:
              res += [- (max(abs(x[k]) - penalty, 0))]
          else:
              res += [0]
      return res

  def soft_threshold_2(x, penalty):
      numrows = x.shape[0]
      numcols = x.shape[1]
      res = np.zeros((numrows, numcols))

      for k in range(numrows):
          for j in range(numcols):
              if x[k][j] > 0:
                  res[k][j] = max(abs(x[k][j]) - penalty, 0)
              elif x[k][j] < 0:
                  res[k][j] = - (max(abs(x[k][j]) - penalty, 0))
              else:
                  res[k][j] = 0
      return res

  def l1norm(x):
      return np.sum(np.abs(x))

  def rpca(ts, lags=7, autodiff=True, is_force_diff=False, s_penalty_factor=0, pvalue_threshold=0.1):

      def compute_l(mu_):
          l_penalty = lpenalty * mu_
          x_temp = input2d - s
          u, singular, v = np.linalg.svd(x_temp, full_matrices=False)
          penalized_d = soft_threshold(singular, l_penalty)
          d_matrix = np.zeros((len(penalized_d), len(penalized_d)))
          np.fill_diagonal(d_matrix, penalized_d)
          l = np.dot(np.dot(u, d_matrix), v)
          return (sum(penalized_d) * l_penalty), l

      def compute_s(mu_):
          s_penalty = s_penalty_factor * mu_
          x_temp = input2d - l
          penalized_s = soft_threshold_2(x_temp, s_penalty)
          s_ = penalized_s
          return ((l1norm(penalized_s.flatten()) * s_penalty), s_)

      def compute_objective(nuclearnorm, l1norm_, l2norm_):
          return 0.5 * l2norm_ + nuclearnorm + l1norm_

      def compute_dynamic_mu():
          m = len(e)
          n = len(input2d[0])
          e_sd = np.std(e.flatten())
          mu = e_sd * math.sqrt(2 * max(m, n))

          return max(.01, mu)

      max_iters = 100
      lpenalty = 1

      residual_len = len(ts) % lags

      ts = ts[residual_len:]
      num_nonzero_records = np.sum(ts != 0)

      nrows = lags
      ncols = int(len(ts) / lags)
      min_nonzero = 3 * nrows

      # set default value for s_penalty_factor if not specified
      if s_penalty_factor == 0:
          s_penalty_factor = 1.4 / float(math.sqrt(max(nrows, ncols)))

      if num_nonzero_records >= min_nonzero:
          # call DickeyFullerTest
          lag_order = math.trunc((len(ts) - 1.0) ** (1.0 / 3.0))  # default in R
          p_value = stattools.adfuller(ts, maxlag=lag_order, autolag=None, regression='ct')[1]
          dickey_needs_diff = False
          if p_value > pvalue_threshold:
              dickey_needs_diff = True

          zero_padded_diff = np.diff(ts)
          zero_padded_diff = np.insert(zero_padded_diff, 0, 0)

          ts_transformed = ts

          if autodiff and dickey_needs_diff:
              # Auto Diff
              ts_transformed = zero_padded_diff
          elif is_force_diff:
              ts_transformed = zero_padded_diff

          # calc mean and std:
          mean = np.mean(ts_transformed)
          stdev = np.std(ts_transformed)

          ts_transformed = np.asarray(ts_transformed)
          ts_transformed = (ts_transformed - mean) / stdev

          input2d = vector_to_matrix(np.array(ts_transformed), nrows, ncols)

          mu = ncols * nrows / float(4 * l1norm(ts_transformed))

          obj_prev = 0.5 * (np.linalg.norm(input2d) ** 2.0)

          tol = 1e-8 * obj_prev
          iter_ = 0
          diff = 2 * tol
          l = np.zeros((nrows, ncols))
          s = np.zeros((nrows, ncols))
          e = np.zeros((nrows, ncols))
          while diff > tol and iter_ < max_iters:
              nuclear_norm, s = compute_s(mu)
              l1_norm, l = compute_l(mu)

              e = input2d - l - s
              norm = np.linalg.norm(e)
              l2_norm = norm ** 2.0

              obj = compute_objective(nuclear_norm, l1_norm, l2_norm)
              diff = abs(obj_prev - obj)
              obj_prev = obj
              mu = compute_dynamic_mu()
              iter_ += iter_

          output_s = s

          for i1 in range(nrows):
              for j1 in range(ncols):
                  output_s[i1][j1] = output_s[i1][j1] * stdev
                  # convert to 1 (outlier), 0 (normal):
                  if (output_s[i1][j1]) * (output_s[i1][j1]) < 0.0001:
                      output_s[i1][j1] = 0
                  else:
                      output_s[i1][j1] = 1

          out = []
          for k in range(residual_len):
              out.append(0)
          r, c = np.shape(output_s)
          for k in range(0, c):
              for j in range(0, r):
                  out.append(output_s[j][k])

          return out
      else:
          return [0] * (len(ts))

  # Extract features of the whole Time Series
  
  output_rpca = rpca(df_output[input_col_name], lags=lags, autodiff=False, is_force_diff=False,
                     s_penalty_factor= 2.4 / math.sqrt(window_length / lags))

  df_output[output_col_name] = output_rpca

  # return df_output.loc[df_output[timestamp_col_name] >= time_max]
  return df_output

# COMMAND ----------

# DBTITLE 1,4: Prophet anomaly detection method
def prophet_anomaly(df_input
  , input_col_name
  , output_col_name='anomaly_prophet'
  , timestamp_col_name='TS'
  , weight=1.0):
  """
  Prophet anomaly detection mehod.
    :param df_input: Pandas dataframe with input data
    :param input_col_name: Name of column where anomaly will be detected
    :param output_col_name: Name of column indicating anomaly
    :parm timestamp_col_name: Name of column that contains timestamp value
    :param weight: Method weight when voting between multiple methods takes place
  """
  
  # Make a copy of input dataframe
  df_output = df_input.copy()
  
  obs_per_period = 144
  window_length = len(df_output)

  
  df_output.sort_values(by=timestamp_col_name, inplace=True)
  df_output[output_col_name] = 0.0

  m = Prophet()
  time_max = df_output[timestamp_col_name].max() - timedelta(minutes=60)
  
  periods = len(df_output.loc[df_output[timestamp_col_name] >= time_max])
  if window_length < 2 * obs_per_period:
    df_output[output_col_name] = 42
    return df_output.loc[df_output[timestamp_col_name] >= time_max]
  
  df = df_output.loc[df_output[timestamp_col_name] < time_max][[input_col_name, timestamp_col_name]].rename(columns={input_col_name: 'y', timestamp_col_name : 'ds'}).reset_index(drop=True)
  m.fit(df)
  future = m.make_future_dataframe(periods=periods)
  forecast = m.predict(future)
  df_output[output_col_name] = 0.0
  df_output.loc[(df_output[timestamp_col_name] >= time_max), output_col_name] = (abs(df_output.loc[(df_output[timestamp_col_name] >= time_max)][input_col_name].values - forecast.yhat.values[:periods]) > 1.5 * abs(forecast.yhat_upper.values[:periods] - forecast.yhat_lower.values[:periods]))
  df_output[output_col_name] = df_output[output_col_name].astype(int)
  #logger.info('Processed %s, a Time-Series %d long in %0.3f secs.' % (X['context'].values[0], df_input.shape[0], time.time() - th))
  
  # return df_output.loc[df_output[timestamp_col_name] >= time_max]
  return df_output

# COMMAND ----------

# DBTITLE 1,Method application and voting
# Define dictionary of available methods with:
#   - method_name: Anomaly detection method name
#   - implementation: Anmaly detection method implementation fuction
#   - weight: Anomaly detection ethod weight - weighted average will be computed based on this figure
#   - ignore: To ignore this anomaly detection method or not

anomaly_detection_methods = {
  'rolling_mad': {"method_name": "rolling_mad", "implementation": rolling_mad_anomaly, "weight": 1.0, "ignore": False}
  , 'twitter': {"method_name": "twitter", "implementation": twitter_anomaly, "weight": 1.0, "ignore": False}
  , 'rpca': {"method_name": "rpca", "implementation": rpca_anomaly, "weight": 1.0, "ignore": False}
  , 'prophet': {"method_name": "prophet", "implementation": prophet_anomaly, "weight": 1.0, "ignore": True}
  , 'arima': {"method_name": "arima", "implementation": None, "weight": 1.0, "ignore": True}
}


# Clean the dictionary and remove methods that are blacklisted or not implemented
for method_name, method in anomaly_detection_methods.copy().items():
    if method.get('ignore') or method.get('implementation') is None:
      del anomaly_detection_methods[method_name]


def anomaly_detection_wrapper(df):
  """
  Wrap all available anomaly detection methods and apply them to given dataframe
  """
  
  # Copy the input DF
  df_output = df.copy()
  
  if len(anomaly_detection_methods) < 1:
    raise Exception('No anomaly detection methods available')
  
  #Iterate over columns
  for col in anomaly_detection_columns:
    # Create summary column and set it to 0
    df_output[f'anomaly_{col}_summary'] = 0
    total_weight = 0
    #Iterate over available methods
    for method_name, method in anomaly_detection_methods.items():
      # Get method result
      print(f'DF has {len(df_output)} rows')
      print(f'Applying method: {method_name}')
      df_output = method['implementation'](df_output, col, f'anomaly_{col}_{method_name}')
      print(f'DF has {len(df_output)} rows')
      # Increase current value of summary by method result * method weight
      df_output[f'anomaly_{col}_summary'] = df_output[f'anomaly_{col}_summary'] + df_output[f'anomaly_{col}_{method_name}'] * method['weight']
      total_weight = total_weight + method['weight']
    #Get sumary per each column
    df_output[f'anomaly_{col}_summary'] = df_output[f'anomaly_{col}_summary'] / total_weight
  
  return df_output

cols_to_select = [ts_column_name, 'TS'] + anomaly_detection_columns

# Prepare string representation of target schema
# First timestamp columns
output_col_info = [f'{ts_column_name} timestamp']

# Second TS which is copied from timestamp column
output_col_info.append('TS timestamp')

# Next all of the columns where anomalies will be detected
for col in anomaly_detection_columns:
  output_col_info.append(f'{col} float')
  
# Finally column with summary and method name suffix
for col in anomaly_detection_columns:
  output_col_info.append(f'anomaly_{col}_summary float')
  for method_name, method in anomaly_detection_methods.copy().items():
    output_col_info.append(f'anomaly_{col}_{method_name} float')
  
output_col_info_text = ', '.join(output_col_info)



# COMMAND ----------

# DBTITLE 1,Applying methods to our data
# Apply all methods and display the results.

streamingWindowDF = (
  spark
  .readStream
  .format("delta")
  .table(streaming_table_name)
  .withColumn("TS", F.col(ts_column_name))
  .select([F.col(c_name) for c_name in cols_to_select])
  .repartition(1)
  .groupBy(window)
  .applyInPandas(anomaly_detection_wrapper, output_col_info_text)
)



# COMMAND ----------

treamingWindowDF \
 .writeStream \
 .option("checkpointLocation", checkpoint_location) \
 .outputMode("append") \
 .table(output_table_name)

 

# COMMAND ----------

# streamingWindowDF = (
#   spark
#   .read
#   .format("delta")
#   .table(streaming_table_name)
#   .withColumn("TS", F.col(ts_column_name))
#   .select([F.col(c_name) for c_name in cols_to_select])
#   .groupBy(window)
#   .applyInPandas(anomaly_detection_wrapper, output_col_info_text)
#   .write
#   .mode("overwrite")
#   .saveAsTable(output_table_name)
# )