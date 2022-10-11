-- Databricks notebook source
-- MAGIC %md-sandbox
-- MAGIC # Turbine data DLT pipeline
-- MAGIC 
-- MAGIC <b style="color:red">TODO: Make input path parametrized. For now input path is fixed and needs to be manually changed to user home dir.</b>

-- COMMAND ----------

CREATE INCREMENTAL LIVE TABLE turbine_bronze_dlt (
  CONSTRAINT correct_schema EXPECT (_rescued_data IS NULL)
)
COMMENT "raw user data coming from json files ingested in incremental with Auto Loader to support schema inference and evolution"
AS SELECT * FROM cloud_files("dbfs:/user/jakub.augustin@databricks.com/anomaly_detection_sol_acc/landing/", "json", map("cloudFiles.inferColumnTypes" , "true"))

-- COMMAND ----------

CREATE INCREMENTAL LIVE TABLE turbine_silver_dlt (
  --CONSTRAINT timestamp_range EXPECT (timestamp BETWEEN 946684801000 and 1672531201000) ON VIOLATION DROP ROW
)
COMMENT "Cleaned data for analysis."
AS SELECT
    `Gearbox_T1_High_Speed_Shaft_Temperature`
    ,`Gearbox_T3_High_Speed_Shaft_Temperature`
    ,`Gearbox_T1_Intermediate_Speed_Shaft_Temperature`
    ,`Temperature_Gearbox_Bearing_Hollow_Shaft`
    ,`Tower_Acceleration_Normal`
    ,`Gearbox_Oil-2_Temperature`
    ,`Tower_Acceleration_Lateral`
    ,`Temperature_Bearing_A`
    ,`Temperature_Trafo-3`
    ,`Gearbox_T3_Intermediate_Speed_Shaft_Temperature`
    ,`Gearbox_Oil-1_Temperature`
    ,`Gearbox_Oil_Temperature`
    ,`Torque`
    ,`Converter_Control_Unit_Reactive_Power`
    ,`Temperature_Trafo-2`
    ,`Reactive_Power`
    ,`Temperature_Shaft_Bearing-1`
    ,`Gearbox_Distributor_Temperature`
    ,`Moment_D_Filtered`
    ,`Moment_D_Direction`
    ,`N-set_1`
    ,`Operating_State`
    ,`Power_Factor`
    ,`Temperature_Shaft_Bearing-2`
    ,`Temperature_Nacelle`
    ,`Voltage_A-N`
    ,`Temperature_Axis_Box-3`
    ,`Voltage_C-N`
    ,`Temperature_Axis_Box-2`
    ,`Temperature_Axis_Box-1`
    ,`Voltage_B-N`
    ,`Nacelle_Position_Degree`
    ,`Converter_Control_Unit_Voltage`
    ,`Temperature_Battery_Box-3`
    ,`Temperature_Battery_Box-2`
    ,`Temperature_Battery_Box-1`
    ,`Hydraulic_Prepressure`
    ,`Angle_Rotor_Position`
    ,`Temperature_Tower_Base`
    ,`Pitch_Offset-2_Asymmetric_Load_Controller`
    ,`Pitch_Offset_Tower_Feedback`
    ,`Line_Frequency`
    ,`Internal_Power_Limit`
    ,`Circuit_Breaker_cut-ins`
    ,`Particle_Counter`
    ,`Tower_Accelaration_Normal_Raw`
    ,`Torque_Offset_Tower_Feedback`
    ,`External_Power_Limit`
    ,`Blade-2_Actual_Value_Angle-B`
    ,`Blade-1_Actual_Value_Angle-B`
    ,`Blade-3_Actual_Value_Angle-B`
    ,`Temperature_Heat_Exchanger_Converter_Control_Unit`
    ,`Tower_Accelaration_Lateral_Raw`
    ,`Temperature_Ambient`
    ,`Nacelle_Revolution`
    ,`Pitch_Offset-1_Asymmetric_Load_Controller`
    ,`Tower_Deflection`
    ,`Pitch_Offset-3_Asymmetric_Load_Controller`
    ,`Wind_Deviation_1_seconds`
    ,`Wind_Deviation_10_seconds`
    ,`Proxy_Sensor_Degree-135`
    ,`State_and_Fault`
    ,`Proxy_Sensor_Degree-225`
    ,`Blade-3_Actual_Value_Angle-A`
    ,`Scope_CH_4`
    ,`Blade-2_Actual_Value_Angle-A`
    ,`Blade-1_Actual_Value_Angle-A`
    ,`Blade-2_Set_Value_Degree`
    ,`Pitch_Demand_Baseline_Degree`
    ,`Blade-1_Set_Value_Degree`
    ,`Blade-3_Set_Value_Degree`
    ,`Moment_Q_Direction`
    ,`Moment_Q_Filltered`
    ,`Proxy_Sensor_Degree-45`
    ,`Turbine_State`
    ,`Proxy_Sensor_Degree-315`
    , cast(`timestamp` as timestamp) as `timestamp`
  from STREAM(live.turbine_bronze_dlt)

-- COMMAND ----------

CREATE INCREMENTAL LIVE TABLE turbine_gold_dlt 
  COMMENT "Final sensor table with all information for Analysis / ML"
AS SELECT * FROM STREAM(live.turbine_silver_dlt)