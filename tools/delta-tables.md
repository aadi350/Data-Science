# A Hands-On Introduction to Delta Tables in Python 

Delta is a versioned parquet file with a transaction log, we get compression and all benefits of parquet  
Transaction log is single-source of truth, see who does what, plays nicely with spark and python  


Python APIs for Delta-Lake
- `pyspark`
- `delta-rs` pip install delta-lake
- pyspark declarative `pip install delta-spark`



```python
from delta import *
from pyspark.sql import SparkSession

builder = SparkSession.builder.appName('delta-tutorial').config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension").config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")

spark = configure_spark_with_delta_pip(builder).getOrCreate()
```

    23/06/24 08:57:56 WARN Utils: Your hostname, debian resolves to a loopback address: 127.0.1.1; using 192.168.100.237 instead (on interface wlp5s0)
    23/06/24 08:57:56 WARN Utils: Set SPARK_LOCAL_IP if you need to bind to another address
    :: loading settings :: url = jar:file:/home/aadi/miniconda3/envs/spark_env/lib/python3.8/site-packages/pyspark/jars/ivy-2.5.1.jar!/org/apache/ivy/core/settings/ivysettings.xml


    Ivy Default Cache set to: /home/aadi/.ivy2/cache
    The jars for the packages stored in: /home/aadi/.ivy2/jars
    io.delta#delta-core_2.12 added as a dependency
    :: resolving dependencies :: org.apache.spark#spark-submit-parent-4c5d06b1-2358-45b9-9407-8a49dcb9bd07;1.0
    	confs: [default]
    	found io.delta#delta-core_2.12;2.3.0 in central
    	found io.delta#delta-storage;2.3.0 in central
    	found org.antlr#antlr4-runtime;4.8 in central
    :: resolution report :: resolve 111ms :: artifacts dl 6ms
    	:: modules in use:
    	io.delta#delta-core_2.12;2.3.0 from central in [default]
    	io.delta#delta-storage;2.3.0 from central in [default]
    	org.antlr#antlr4-runtime;4.8 from central in [default]
    	---------------------------------------------------------------------
    	|                  |            modules            ||   artifacts   |
    	|       conf       | number| search|dwnlded|evicted|| number|dwnlded|
    	---------------------------------------------------------------------
    	|      default     |   3   |   0   |   0   |   0   ||   3   |   0   |
    	---------------------------------------------------------------------
    :: retrieving :: org.apache.spark#spark-submit-parent-4c5d06b1-2358-45b9-9407-8a49dcb9bd07
    	confs: [default]
    	0 artifacts copied, 3 already retrieved (0kB/4ms)


    23/06/24 08:57:57 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable


    Setting default log level to "WARN".
    To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).



```python
# load vanilla spark table
sdf = spark.read.load(
  '/storage/data/airline_2m.csv' ,
  format='com.databricks.spark.csv',
  header='true',
  inferSchema='true'
).select(['FlightDate', 'Reporting_Airline', 'Flight_Number_Reporting_Airline','Origin', 'Dest', 'DepTime', 'DepDelay', 'ArrTime', 'ArrDelay' ])

# save as a delta table
sdf.write.format('delta').mode('overwrite').save('/storage/data/airline_2m.delta')
```

                                                                                    

    23/06/10 12:36:43 WARN MemoryManager: Total allocation exceeds 95.00% (1,012,583,616 bytes) of heap memory
    Scaling row group sizes to 94.30% for 8 writers
    23/06/10 12:36:43 WARN MemoryManager: Total allocation exceeds 95.00% (1,012,583,616 bytes) of heap memory
    Scaling row group sizes to 83.83% for 9 writers
    23/06/10 12:36:43 WARN MemoryManager: Total allocation exceeds 95.00% (1,012,583,616 bytes) of heap memory
    Scaling row group sizes to 75.44% for 10 writers
    23/06/10 12:36:43 WARN MemoryManager: Total allocation exceeds 95.00% (1,012,583,616 bytes) of heap memory
    Scaling row group sizes to 68.58% for 11 writers
    23/06/10 12:36:43 WARN MemoryManager: Total allocation exceeds 95.00% (1,012,583,616 bytes) of heap memory
    Scaling row group sizes to 62.87% for 12 writers


    [Stage 178:>                                                      (0 + 12) / 12]

    23/06/10 12:36:44 WARN MemoryManager: Total allocation exceeds 95.00% (1,012,583,616 bytes) of heap memory
    Scaling row group sizes to 68.58% for 11 writers
    23/06/10 12:36:45 WARN MemoryManager: Total allocation exceeds 95.00% (1,012,583,616 bytes) of heap memory
    Scaling row group sizes to 75.44% for 10 writers
    23/06/10 12:36:45 WARN MemoryManager: Total allocation exceeds 95.00% (1,012,583,616 bytes) of heap memory
    Scaling row group sizes to 83.83% for 9 writers
    23/06/10 12:36:45 WARN MemoryManager: Total allocation exceeds 95.00% (1,012,583,616 bytes) of heap memory
    Scaling row group sizes to 94.30% for 8 writers


                                                                                    


```python
delta_table = DeltaTable.forPath(spark, '/storage/data/airline_2m.delta/')
```


```python
# inspecting what we just wrote
import os

os.listdir('/storage/data/airline_2m.delta/')
# bunch of individual files and a folder called "_delta_log"
```




    ['part-00008-37835621-905f-481b-bee1-f2fbde02595c-c000.snappy.parquet',
     '.part-00002-f79474ae-503d-4914-8aff-54af17da6de2-c000.snappy.parquet.crc',
     '.part-00006-9cdf358f-5ff7-4c45-9314-09aeeb770ab4-c000.snappy.parquet.crc',
     '.part-00001-da20e657-cbd5-4ca0-b7bf-f862a6083d2b-c000.snappy.parquet.crc',
     'part-00007-3e4232fc-fbd4-4cce-84fb-7a862aff9928-c000.snappy.parquet',
     '.part-00010-bff9f938-783e-490b-b462-636415a9f94d-c000.snappy.parquet.crc',
     '.part-00009-6c30192d-45cc-4819-b3ce-3e6c4de1c218-c000.snappy.parquet.crc',
     'part-00004-016a4d0d-6026-4cff-9c53-8a4af7c32241-c000.snappy.parquet',
     'part-00005-3885ce51-666c-4fb2-a0ad-a12c90ec095a-c000.snappy.parquet',
     '_delta_log',
     'part-00003-9aec4e25-09dd-4f1d-8bff-e6c7c368ae7d-c000.snappy.parquet',
     '.part-00008-37835621-905f-481b-bee1-f2fbde02595c-c000.snappy.parquet.crc',
     'part-00006-9cdf358f-5ff7-4c45-9314-09aeeb770ab4-c000.snappy.parquet',
     'part-00009-6c30192d-45cc-4819-b3ce-3e6c4de1c218-c000.snappy.parquet',
     '.part-00003-9aec4e25-09dd-4f1d-8bff-e6c7c368ae7d-c000.snappy.parquet.crc',
     'part-00002-f79474ae-503d-4914-8aff-54af17da6de2-c000.snappy.parquet',
     'part-00010-bff9f938-783e-490b-b462-636415a9f94d-c000.snappy.parquet',
     'part-00011-eeea5de3-5e91-47df-9762-7d673770f769-c000.snappy.parquet',
     '.part-00011-eeea5de3-5e91-47df-9762-7d673770f769-c000.snappy.parquet.crc',
     '.part-00007-3e4232fc-fbd4-4cce-84fb-7a862aff9928-c000.snappy.parquet.crc',
     '.part-00005-3885ce51-666c-4fb2-a0ad-a12c90ec095a-c000.snappy.parquet.crc',
     'part-00001-da20e657-cbd5-4ca0-b7bf-f862a6083d2b-c000.snappy.parquet',
     '.part-00000-9642bf76-d935-42ae-bbe4-64d7b1c52df2-c000.snappy.parquet.crc',
     'part-00000-9642bf76-d935-42ae-bbe4-64d7b1c52df2-c000.snappy.parquet',
     '.part-00004-016a4d0d-6026-4cff-9c53-8a4af7c32241-c000.snappy.parquet.crc']




```python
# _delta_log
os.listdir('/storage/data/airline_2m.delta/_delta_log/')
# a json and a .crc file, crc files are checksums added to prevent corruption if parquet is corrupted in-flight
```




    ['00000000000000000000.json', '.00000000000000000000.json.crc']



logs hold:
copy-paste:
Whenever a user performs an operation to modify a table (such as an INSERT, UPDATE or DELETE), Delta Lake breaks that operation down into a series of discrete steps composed of one or more of the actions below.

Add file - adds a data file.
Remove file - removes a data file.
Update metadata - Updates the table’s metadata (e.g., changing the table’s name, schema or partitioning).
Set transaction - Records that a structured streaming job has committed a micro-batch with the given ID.
Change protocol - enables new features by switching the Delta Lake transaction log to the newest software protocol.
Commit info - Contains information around the commit, which operation was made, from where and at what time.


WHen table is created, table's transaciton log automatically created in `_delta_log`. Each change is recorded as an atomic commit in the transaction log

Time-travel `DESCRIBE HISTORY` in SQL, can select fomr using `TIMESTAMP` OR the `VERSION` `TABLE@v2` 


Delta transaciton enables ACID, full audit and scalable metadata.

Hive metastore stores table definition, where table is stored


# CTAS Statements
`CREATE_TABLE _ AS SELECT` use output of select to crae

Does not support manual schema declaration, automatically infers schema from query results, does not require an `INSERT` statement.

```CRETE TABLE new_table
COMMENT "some comment"
PARTITIONED BY (id1, id2) --best practice to non-partition
LOCATION '/some/path'
FROM ...
```

# Constraints
- NOT NULL and CHECK supported
`CHECK` looks like `WHERE` clauses, `NOT NULL` is obvious

# Copying
**DEEP** clone  
`CREATE TABLE table_clone DEEP CLONE source_table` can occur incrementall (done multiple times)

**SHALLOW** clone  
only copies transaction log, no actual data copied  


# Things to observe 
- Updates
- Delete
- Optimize: compacts multiple small files
- ZORDER BY: added to optimize, this compacts files ordered by the column, like indexing
    - Zordering for high-cardinality columns (>= 2 columns): can I go ahead and skip (i.e. reduce number of files need to scan), diminishing returns if z-order by all columns
    - Partitioning is for low-cardinality columns, table >= 1TB of data, partitions >= 1GB
- Cleaning up: delta lake `VACUUM table_name [retention period]`, feault retention period is 7 days. Once vacuum is run, time-travel is lost
- Try deleting then using time-travel to go back using `RESTORE TABLE` then calling `DESCRIBE HISTORY`  

can call `delta_table.history().show()`  


```python
import json

with open('/storage/data/airline_2m.delta/_delta_log/00000000000000000000.json', 'r') as json_file:
    for line in json_file:
      json_object = json.loads(line) 
      print(json.dumps(json_object, indent=2))
  
```

    {
      "commitInfo": {
        "timestamp": 1686413205823,
        "operation": "WRITE",
        "operationParameters": {
          "mode": "Overwrite",
          "partitionBy": "[]"
        },
        "isolationLevel": "Serializable",
        "isBlindAppend": false,
        "operationMetrics": {
          "numFiles": "12",
          "numOutputRows": "2000000",
          "numOutputBytes": "20619620"
        },
        "engineInfo": "Apache-Spark/3.3.2 Delta-Lake/2.3.0",
        "txnId": "4f5656f9-d9d2-404e-bf11-722a1349387b"
      }
    }
    {
      "protocol": {
        "minReaderVersion": 1,
        "minWriterVersion": 2
      }
    }
    {
      "metaData": {
        "id": "609a6c9c-5c5d-4844-9276-49201985b8f1",
        "format": {
          "provider": "parquet",
          "options": {}
        },
        "schemaString": "{\"type\":\"struct\",\"fields\":[{\"name\":\"FlightDate\",\"type\":\"timestamp\",\"nullable\":true,\"metadata\":{}},{\"name\":\"Reporting_Airline\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}},{\"name\":\"Origin\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}},{\"name\":\"Dest\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}},{\"name\":\"DepTime\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"DepDelay\",\"type\":\"double\",\"nullable\":true,\"metadata\":{}},{\"name\":\"ArrTime\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"ArrDelay\",\"type\":\"double\",\"nullable\":true,\"metadata\":{}}]}",
        "partitionColumns": [],
        "configuration": {},
        "createdTime": 1686413204081
      }
    }
    {
      "add": {
        "path": "part-00000-9642bf76-d935-42ae-bbe4-64d7b1c52df2-c000.snappy.parquet",
        "partitionValues": {},
        "size": 1726172,
        "modificationTime": 1686413205702,
        "dataChange": true,
        "stats": "{\"numRecords\":167450,\"minValues\":{\"FlightDate\":\"1987-10-01T00:00:00.000-04:00\",\"Reporting_Airline\":\"9E\",\"Origin\":\"ABE\",\"Dest\":\"ABE\",\"DepTime\":1,\"DepDelay\":-60.0,\"ArrTime\":1,\"ArrDelay\":-89.0},\"maxValues\":{\"FlightDate\":\"2020-03-31T00:00:00.000-04:00\",\"Reporting_Airline\":\"YX\",\"Origin\":\"YUM\",\"Dest\":\"YUM\",\"DepTime\":2400,\"DepDelay\":1435.0,\"ArrTime\":2400,\"ArrDelay\":1153.0},\"nullCount\":{\"FlightDate\":0,\"Reporting_Airline\":0,\"Origin\":0,\"Dest\":0,\"DepTime\":3012,\"DepDelay\":3016,\"ArrTime\":3309,\"ArrDelay\":3446}}"
      }
    }
    {
      "add": {
        "path": "part-00001-da20e657-cbd5-4ca0-b7bf-f862a6083d2b-c000.snappy.parquet",
        "partitionValues": {},
        "size": 1726016,
        "modificationTime": 1686413205734,
        "dataChange": true,
        "stats": "{\"numRecords\":167446,\"minValues\":{\"FlightDate\":\"1987-10-01T00:00:00.000-04:00\",\"Reporting_Airline\":\"9E\",\"Origin\":\"ABE\",\"Dest\":\"ABE\",\"DepTime\":1,\"DepDelay\":-82.0,\"ArrTime\":1,\"ArrDelay\":-80.0},\"maxValues\":{\"FlightDate\":\"2020-03-31T00:00:00.000-04:00\",\"Reporting_Airline\":\"YX\",\"Origin\":\"YUM\",\"Dest\":\"YUM\",\"DepTime\":2400,\"DepDelay\":1434.0,\"ArrTime\":2400,\"ArrDelay\":1295.0},\"nullCount\":{\"FlightDate\":0,\"Reporting_Airline\":0,\"Origin\":0,\"Dest\":0,\"DepTime\":2997,\"DepDelay\":3005,\"ArrTime\":3307,\"ArrDelay\":3436}}"
      }
    }
    {
      "add": {
        "path": "part-00002-f79474ae-503d-4914-8aff-54af17da6de2-c000.snappy.parquet",
        "partitionValues": {},
        "size": 1725906,
        "modificationTime": 1686413205706,
        "dataChange": true,
        "stats": "{\"numRecords\":167449,\"minValues\":{\"FlightDate\":\"1987-10-01T00:00:00.000-04:00\",\"Reporting_Airline\":\"9E\",\"Origin\":\"ABE\",\"Dest\":\"ABE\",\"DepTime\":1,\"DepDelay\":-62.0,\"ArrTime\":1,\"ArrDelay\":-90.0},\"maxValues\":{\"FlightDate\":\"2020-03-31T00:00:00.000-04:00\",\"Reporting_Airline\":\"YX\",\"Origin\":\"YUM\",\"Dest\":\"YUM\",\"DepTime\":2400,\"DepDelay\":1855.0,\"ArrTime\":2400,\"ArrDelay\":1847.0},\"nullCount\":{\"FlightDate\":0,\"Reporting_Airline\":0,\"Origin\":0,\"Dest\":0,\"DepTime\":2998,\"DepDelay\":3009,\"ArrTime\":3304,\"ArrDelay\":3437}}"
      }
    }
    {
      "add": {
        "path": "part-00003-9aec4e25-09dd-4f1d-8bff-e6c7c368ae7d-c000.snappy.parquet",
        "partitionValues": {},
        "size": 1726098,
        "modificationTime": 1686413205622,
        "dataChange": true,
        "stats": "{\"numRecords\":167467,\"minValues\":{\"FlightDate\":\"1987-10-01T00:00:00.000-04:00\",\"Reporting_Airline\":\"9E\",\"Origin\":\"ABE\",\"Dest\":\"ABE\",\"DepTime\":1,\"DepDelay\":-67.0,\"ArrTime\":1,\"ArrDelay\":-81.0},\"maxValues\":{\"FlightDate\":\"2020-03-31T00:00:00.000-04:00\",\"Reporting_Airline\":\"YX\",\"Origin\":\"YUM\",\"Dest\":\"YUM\",\"DepTime\":2400,\"DepDelay\":1878.0,\"ArrTime\":2400,\"ArrDelay\":1898.0},\"nullCount\":{\"FlightDate\":0,\"Reporting_Airline\":0,\"Origin\":0,\"Dest\":0,\"DepTime\":2989,\"DepDelay\":2994,\"ArrTime\":3252,\"ArrDelay\":3388}}"
      }
    }
    {
      "add": {
        "path": "part-00004-016a4d0d-6026-4cff-9c53-8a4af7c32241-c000.snappy.parquet",
        "partitionValues": {},
        "size": 1726276,
        "modificationTime": 1686413205718,
        "dataChange": true,
        "stats": "{\"numRecords\":167462,\"minValues\":{\"FlightDate\":\"1987-10-01T00:00:00.000-04:00\",\"Reporting_Airline\":\"9E\",\"Origin\":\"ABE\",\"Dest\":\"ABE\",\"DepTime\":1,\"DepDelay\":-60.0,\"ArrTime\":1,\"ArrDelay\":-77.0},\"maxValues\":{\"FlightDate\":\"2020-03-31T00:00:00.000-04:00\",\"Reporting_Airline\":\"YX\",\"Origin\":\"YUM\",\"Dest\":\"YUM\",\"DepTime\":2400,\"DepDelay\":1628.0,\"ArrTime\":2400,\"ArrDelay\":1631.0},\"nullCount\":{\"FlightDate\":0,\"Reporting_Airline\":0,\"Origin\":0,\"Dest\":0,\"DepTime\":3042,\"DepDelay\":3044,\"ArrTime\":3325,\"ArrDelay\":3474}}"
      }
    }
    {
      "add": {
        "path": "part-00005-3885ce51-666c-4fb2-a0ad-a12c90ec095a-c000.snappy.parquet",
        "partitionValues": {},
        "size": 1727251,
        "modificationTime": 1686413205710,
        "dataChange": true,
        "stats": "{\"numRecords\":167481,\"minValues\":{\"FlightDate\":\"1987-10-01T00:00:00.000-04:00\",\"Reporting_Airline\":\"9E\",\"Origin\":\"ABE\",\"Dest\":\"ABE\",\"DepTime\":1,\"DepDelay\":-49.0,\"ArrTime\":1,\"ArrDelay\":-90.0},\"maxValues\":{\"FlightDate\":\"2020-03-31T00:00:00.000-04:00\",\"Reporting_Airline\":\"YX\",\"Origin\":\"YUM\",\"Dest\":\"YUM\",\"DepTime\":2400,\"DepDelay\":1435.0,\"ArrTime\":2400,\"ArrDelay\":1458.0},\"nullCount\":{\"FlightDate\":0,\"Reporting_Airline\":0,\"Origin\":0,\"Dest\":0,\"DepTime\":3059,\"DepDelay\":3066,\"ArrTime\":3380,\"ArrDelay\":3519}}"
      }
    }
    {
      "add": {
        "path": "part-00006-9cdf358f-5ff7-4c45-9314-09aeeb770ab4-c000.snappy.parquet",
        "partitionValues": {},
        "size": 1726159,
        "modificationTime": 1686413205670,
        "dataChange": true,
        "stats": "{\"numRecords\":167477,\"minValues\":{\"FlightDate\":\"1987-10-01T00:00:00.000-04:00\",\"Reporting_Airline\":\"9E\",\"Origin\":\"ABE\",\"Dest\":\"ABE\",\"DepTime\":1,\"DepDelay\":-62.0,\"ArrTime\":1,\"ArrDelay\":-76.0},\"maxValues\":{\"FlightDate\":\"2020-03-31T00:00:00.000-04:00\",\"Reporting_Airline\":\"YX\",\"Origin\":\"YUM\",\"Dest\":\"YUM\",\"DepTime\":2400,\"DepDelay\":1430.0,\"ArrTime\":2400,\"ArrDelay\":1402.0},\"nullCount\":{\"FlightDate\":0,\"Reporting_Airline\":0,\"Origin\":0,\"Dest\":0,\"DepTime\":3069,\"DepDelay\":3073,\"ArrTime\":3357,\"ArrDelay\":3470}}"
      }
    }
    {
      "add": {
        "path": "part-00007-3e4232fc-fbd4-4cce-84fb-7a862aff9928-c000.snappy.parquet",
        "partitionValues": {},
        "size": 1725890,
        "modificationTime": 1686413205702,
        "dataChange": true,
        "stats": "{\"numRecords\":167445,\"minValues\":{\"FlightDate\":\"1987-10-01T00:00:00.000-04:00\",\"Reporting_Airline\":\"9E\",\"Origin\":\"ABE\",\"Dest\":\"ABE\",\"DepTime\":1,\"DepDelay\":-60.0,\"ArrTime\":1,\"ArrDelay\":-91.0},\"maxValues\":{\"FlightDate\":\"2020-03-31T00:00:00.000-04:00\",\"Reporting_Airline\":\"YX\",\"Origin\":\"YUM\",\"Dest\":\"YUM\",\"DepTime\":2400,\"DepDelay\":1538.0,\"ArrTime\":2400,\"ArrDelay\":1532.0},\"nullCount\":{\"FlightDate\":0,\"Reporting_Airline\":0,\"Origin\":0,\"Dest\":0,\"DepTime\":2989,\"DepDelay\":2994,\"ArrTime\":3301,\"ArrDelay\":3447}}"
      }
    }
    {
      "add": {
        "path": "part-00008-37835621-905f-481b-bee1-f2fbde02595c-c000.snappy.parquet",
        "partitionValues": {},
        "size": 1726165,
        "modificationTime": 1686413205722,
        "dataChange": true,
        "stats": "{\"numRecords\":167451,\"minValues\":{\"FlightDate\":\"1987-10-01T00:00:00.000-04:00\",\"Reporting_Airline\":\"9E\",\"Origin\":\"ABE\",\"Dest\":\"ABE\",\"DepTime\":1,\"DepDelay\":-489.0,\"ArrTime\":1,\"ArrDelay\":-480.0},\"maxValues\":{\"FlightDate\":\"2020-03-31T00:00:00.000-04:00\",\"Reporting_Airline\":\"YX\",\"Origin\":\"YUM\",\"Dest\":\"YUM\",\"DepTime\":2400,\"DepDelay\":1437.0,\"ArrTime\":2400,\"ArrDelay\":1343.0},\"nullCount\":{\"FlightDate\":0,\"Reporting_Airline\":0,\"Origin\":0,\"Dest\":0,\"DepTime\":3031,\"DepDelay\":3035,\"ArrTime\":3356,\"ArrDelay\":3475}}"
      }
    }
    {
      "add": {
        "path": "part-00009-6c30192d-45cc-4819-b3ce-3e6c4de1c218-c000.snappy.parquet",
        "partitionValues": {},
        "size": 1725812,
        "modificationTime": 1686413205670,
        "dataChange": true,
        "stats": "{\"numRecords\":167448,\"minValues\":{\"FlightDate\":\"1987-10-01T00:00:00.000-04:00\",\"Reporting_Airline\":\"9E\",\"Origin\":\"ABE\",\"Dest\":\"ABE\",\"DepTime\":1,\"DepDelay\":-990.0,\"ArrTime\":1,\"ArrDelay\":-706.0},\"maxValues\":{\"FlightDate\":\"2020-03-31T00:00:00.000-04:00\",\"Reporting_Airline\":\"YX\",\"Origin\":\"YUM\",\"Dest\":\"YUM\",\"DepTime\":2400,\"DepDelay\":1435.0,\"ArrTime\":2400,\"ArrDelay\":1430.0},\"nullCount\":{\"FlightDate\":0,\"Reporting_Airline\":0,\"Origin\":0,\"Dest\":0,\"DepTime\":2996,\"DepDelay\":2999,\"ArrTime\":3289,\"ArrDelay\":3416}}"
      }
    }
    {
      "add": {
        "path": "part-00010-bff9f938-783e-490b-b462-636415a9f94d-c000.snappy.parquet",
        "partitionValues": {},
        "size": 1726428,
        "modificationTime": 1686413205698,
        "dataChange": true,
        "stats": "{\"numRecords\":167477,\"minValues\":{\"FlightDate\":\"1987-10-01T00:00:00.000-04:00\",\"Reporting_Airline\":\"9E\",\"Origin\":\"ABE\",\"Dest\":\"ABE\",\"DepTime\":1,\"DepDelay\":-43.0,\"ArrTime\":1,\"ArrDelay\":-94.0},\"maxValues\":{\"FlightDate\":\"2020-03-31T00:00:00.000-04:00\",\"Reporting_Airline\":\"YX\",\"Origin\":\"YUM\",\"Dest\":\"YUM\",\"DepTime\":2400,\"DepDelay\":1435.0,\"ArrTime\":2400,\"ArrDelay\":1467.0},\"nullCount\":{\"FlightDate\":0,\"Reporting_Airline\":0,\"Origin\":0,\"Dest\":0,\"DepTime\":3017,\"DepDelay\":3021,\"ArrTime\":3289,\"ArrDelay\":3388}}"
      }
    }
    {
      "add": {
        "path": "part-00011-eeea5de3-5e91-47df-9762-7d673770f769-c000.snappy.parquet",
        "partitionValues": {},
        "size": 1631447,
        "modificationTime": 1686413205710,
        "dataChange": true,
        "stats": "{\"numRecords\":157947,\"minValues\":{\"FlightDate\":\"1987-10-01T00:00:00.000-04:00\",\"Reporting_Airline\":\"9E\",\"Origin\":\"ABE\",\"Dest\":\"ABE\",\"DepTime\":1,\"DepDelay\":-63.0,\"ArrTime\":1,\"ArrDelay\":-81.0},\"maxValues\":{\"FlightDate\":\"2020-03-31T00:00:00.000-04:00\",\"Reporting_Airline\":\"YX\",\"Origin\":\"YUM\",\"Dest\":\"YUM\",\"DepTime\":2400,\"DepDelay\":1434.0,\"ArrTime\":2400,\"ArrDelay\":1007.0},\"nullCount\":{\"FlightDate\":0,\"Reporting_Airline\":0,\"Origin\":0,\"Dest\":0,\"DepTime\":2806,\"DepDelay\":2812,\"ArrTime\":3082,\"ArrDelay\":3182}}"
      }
    }



```python
sdf.filter('Origin="JFK" and FlightDate = "2017-07-26"').show() # filter for flights leaving JFK on the 26th July, 2017
```

    +-------------------+-----------------+-------------------------------+------+----+-------+--------+-------+--------+
    |         FlightDate|Reporting_Airline|Flight_Number_Reporting_Airline|Origin|Dest|DepTime|DepDelay|ArrTime|ArrDelay|
    +-------------------+-----------------+-------------------------------+------+----+-------+--------+-------+--------+
    |2017-07-26 00:00:00|               AA|                              9|   JFK| SFO|    657|    -3.0|    947|   -33.0|
    |2017-07-26 00:00:00|               DL|                           2051|   JFK| SJU|   2036|     7.0|     17|   -26.0|
    |2017-07-26 00:00:00|               B6|                           1089|   JFK| MCO|   1340|    21.0|   1638|    27.0|
    +-------------------+-----------------+-------------------------------+------+----+-------+--------+-------+--------+
    


                                                                                    


```python
# create new table for upsert
sdf.dtypes

updates = [
    (datetime.strptime('2017-07-26', '%Y-%m-%d'), 'AA',9, 'JFK', 'SFO', 756, -4.0, 1117, 0.0), # update existing entry
    (datetime.strptime('2017-07-26', '%Y-%m-%d'), 'DL',1368, 'JFK', 'MIA', 1107, 1.0, 1421, 0.0), # new entry
]
(updates_table := spark.createDataFrame(updates, schema=sdf.schema)).show()
```

    +-------------------+-----------------+-------------------------------+------+----+-------+--------+-------+--------+
    |         FlightDate|Reporting_Airline|Flight_Number_Reporting_Airline|Origin|Dest|DepTime|DepDelay|ArrTime|ArrDelay|
    +-------------------+-----------------+-------------------------------+------+----+-------+--------+-------+--------+
    |2017-07-26 00:00:00|               AA|                              9|   JFK| SFO|    756|    -4.0|   1117|     0.0|
    |2017-07-26 00:00:00|               DL|                           1368|   JFK| MIA|   1107|     1.0|   1421|     0.0|
    +-------------------+-----------------+-------------------------------+------+----+-------+--------+-------+--------+
    



```python
# perform upsert
(delta_table.alias('current_data')
  .merge(
      source=updates_table.alias('new_data'), 
      condition=F.expr('current_data.FlightDate = new_data.FlightDate and new_data.Origin = current_data.Origin and current_data.Dest = new_data.Dest and  current_data.Reporting_Airline = new_data.Reporting_Airline and current_data.Flight_Number_Reporting_Airline = new_data.Flight_Number_Reporting_Airline'))
  .whenMatchedUpdate(set = {
    'DepTime': F.col('new_data.DepTime'),
    'DepDelay': F.col('new_data.DepDelay'),
    'ArrTime': F.col('new_data.ArrTime'),
    'ArrDelay': F.col('new_data.ArrDelay'),
  })
  # .whenNotMatchedInsert(values = {
  #   'FlightDate': 'new_data.FlightDate',
  #   'Reporting_Airline': 'new_data.Reporting_Airline',
  #   'Flight_Number_Reporting_Airline': 'new_data.Flight_Number_Reporting_Airline',
  #   'DepTime': 'new_data.DepTime',
  #   'DepDelay': 'new_data.DepDelay',
  #   'ArrTime': 'new_data.ArrTime',
  #   'ArrDelay': 'new_data.ArrDelay',
  # })
  .whenNotMatchedInsertAll()
  .execute()

  # can have any number of whenMatched (at most one update and one delete action)
  # update in merge only updates sepcified columns
  # multiple whenMatched executes in order specified
  # to update all columns of target dleta table use whenMatched(...).updateAll()


  # whenNotMatched when source doesn't match target
  # can have ONLY the insert action, any unspecified columns assume null
  # each whenNotMatched can have optional conditoin, 
  # see https://docs.delta.io/latest/delta-update.html#language-python
)
```


```python
delta_table.toDF().filter('Origin="JFK" and FlightDate = "2017-07-26"').show()
```

    +-------------------+-----------------+-------------------------------+------+----+-------+--------+-------+--------+
    |         FlightDate|Reporting_Airline|Flight_Number_Reporting_Airline|Origin|Dest|DepTime|DepDelay|ArrTime|ArrDelay|
    +-------------------+-----------------+-------------------------------+------+----+-------+--------+-------+--------+
    |2017-07-26 00:00:00|               DL|                           2051|   JFK| SJU|   2036|     7.0|     17|   -26.0|
    |2017-07-26 00:00:00|               B6|                           1089|   JFK| MCO|   1340|    21.0|   1638|    27.0|
    |2017-07-26 00:00:00|               DL|                           1368|   JFK| MIA|   1107|     1.0|   1421|     0.0|
    |2017-07-26 00:00:00|               AA|                              9|   JFK| SFO|    756|    -4.0|   1117|     0.0|
    +-------------------+-----------------+-------------------------------+------+----+-------+--------+-------+--------+
    



```python
import pandas as pd

df = pd.DataFrame({
  'a': ['A', 'B', 'C', 'D'],
  'value': [1, 2, 3, 4]
})

sdf = spark.createDataFrame(df)
```


```python
from pyspark.sql import functions as F

sdf.filter(F.col('a').isin(['A', 'B'])).show()
```

    +---+-----+
    |  a|value|
    +---+-----+
    |  A|    1|
    |  B|    2|
    +---+-----+
    


# Content Starts Here


This post covers the Delta Lake, which is an open-source format extending parquet files for ACID transactions. More specifically, this covers how to work with Delta tables using the `pyspark` and native `Delta` APIs. 

Delta tables can be thought of as having the benefits of a non-flat file format (compression via more efficient encoding), with a single source of truth called the transaction log.

## Creating a Delta Table

In order to create a delta table, I'm loading an existing CSV using `pyspark`, and saving it using the `format` option in `pyspark`'s `write`:

(Completely irrelevant, however the dataset being used here is [IBM's Airline Reporting Carrier On-Timer Performance Dataset](https://developer.ibm.com/exchanges/data/all/airline/))


```python
# load original dataset
sdf = spark.read.load(
  '/storage/data/airline_2m.csv' ,
  format='com.databricks.spark.csv',
  header='true',
  inferSchema='true'
).select(['FlightDate', 'Reporting_Airline', 'Flight_Number_Reporting_Airline','Origin', 'Dest', 'DepTime', 'DepDelay', 'ArrTime', 'ArrDelay' ]).filter('Origin="JFK" and FlightDate>="2017-12-01" and FlightDate <= "2017-12-31"')

# write as a delta table
sdf.write.format('delta').mode('overwrite').save('/storage/data/airline_2m.delta')
```

                                                                                    

    23/06/22 22:09:51 WARN MemoryManager: Total allocation exceeds 95.00% (986,185,716 bytes) of heap memory
    Scaling row group sizes to 91.85% for 8 writers
    23/06/22 22:09:51 WARN MemoryManager: Total allocation exceeds 95.00% (986,185,716 bytes) of heap memory
    Scaling row group sizes to 81.64% for 9 writers
    23/06/22 22:09:51 WARN MemoryManager: Total allocation exceeds 95.00% (986,185,716 bytes) of heap memory
    Scaling row group sizes to 73.48% for 10 writers
    23/06/22 22:09:51 WARN MemoryManager: Total allocation exceeds 95.00% (986,185,716 bytes) of heap memory
    Scaling row group sizes to 66.80% for 11 writers


    [Stage 57:>                                                       (0 + 12) / 12]

    23/06/22 22:09:51 WARN MemoryManager: Total allocation exceeds 95.00% (986,185,716 bytes) of heap memory
    Scaling row group sizes to 61.23% for 12 writers
    23/06/22 22:09:52 WARN MemoryManager: Total allocation exceeds 95.00% (986,185,716 bytes) of heap memory
    Scaling row group sizes to 66.80% for 11 writers
    23/06/22 22:09:52 WARN MemoryManager: Total allocation exceeds 95.00% (986,185,716 bytes) of heap memory
    Scaling row group sizes to 73.48% for 10 writers
    23/06/22 22:09:52 WARN MemoryManager: Total allocation exceeds 95.00% (986,185,716 bytes) of heap memory
    Scaling row group sizes to 81.64% for 9 writers
    23/06/22 22:09:52 WARN MemoryManager: Total allocation exceeds 95.00% (986,185,716 bytes) of heap memory
    Scaling row group sizes to 91.85% for 8 writers


                                                                                    

Alternatively, we can use the `CTAS` statement in sql:
(note that this is for the purposes of demonstration and new_table is not used for the rest of this notebook)


```python
spark.sql('''
  CREATE OR REPLACE TABLE new_table
  using DELTA AS SELECT * FROM csv.`/storage/data/airline_2m.csv`
''')
```

    23/06/24 09:18:56 WARN package: Truncated the string representation of a plan since it was too large. This behavior can be adjusted by setting 'spark.sql.debug.maxToStringFields'.


    [Stage 6:>                                                        (0 + 12) / 12]

    23/06/24 09:18:57 WARN MemoryManager: Total allocation exceeds 95.00% (951,320,564 bytes) of heap memory
    Scaling row group sizes to 88.60% for 8 writers
    23/06/24 09:18:57 WARN MemoryManager: Total allocation exceeds 95.00% (951,320,564 bytes) of heap memory
    Scaling row group sizes to 78.75% for 9 writers
    23/06/24 09:18:57 WARN MemoryManager: Total allocation exceeds 95.00% (951,320,564 bytes) of heap memory
    Scaling row group sizes to 70.88% for 10 writers
    23/06/24 09:18:57 WARN MemoryManager: Total allocation exceeds 95.00% (951,320,564 bytes) of heap memory
    Scaling row group sizes to 64.44% for 11 writers
    23/06/24 09:18:57 WARN MemoryManager: Total allocation exceeds 95.00% (951,320,564 bytes) of heap memory
    Scaling row group sizes to 59.07% for 12 writers
    23/06/24 09:19:10 WARN MemoryManager: Total allocation exceeds 95.00% (951,320,564 bytes) of heap memory
    Scaling row group sizes to 64.44% for 11 writers


    [Stage 6:====>                                                    (1 + 11) / 12]

    23/06/24 09:19:10 WARN MemoryManager: Total allocation exceeds 95.00% (951,320,564 bytes) of heap memory
    Scaling row group sizes to 70.88% for 10 writers
    23/06/24 09:19:10 WARN MemoryManager: Total allocation exceeds 95.00% (951,320,564 bytes) of heap memory
    Scaling row group sizes to 78.75% for 9 writers


    [Stage 6:==============>                                           (3 + 9) / 12]

    23/06/24 09:19:10 WARN MemoryManager: Total allocation exceeds 95.00% (951,320,564 bytes) of heap memory
    Scaling row group sizes to 88.60% for 8 writers


                                                                                    




    DataFrame[]



Now, if we inspect the path which was written to, there are two things to note:
1. There are multiple `.parquet` files
2. There's a directory called the `_delta_log`.


```python
%ls /storage/data/airline_2m.delta
```

    [0m[01;34m_delta_log[0m/
    part-00000-9db93c29-a618-4f69-aa5f-776e1ca1a221-c000.snappy.parquet
    part-00001-43218537-207b-4569-8d98-7cb1d2959d3d-c000.snappy.parquet
    part-00002-b41a2670-c5bc-4515-93c6-c9fe87c3d132-c000.snappy.parquet
    part-00003-0393fe9a-e8cc-4c69-83a4-e11828b75886-c000.snappy.parquet
    part-00004-edbda9cf-91b8-4752-bec3-f30e93651fe8-c000.snappy.parquet
    part-00005-9c275ad8-871a-4948-9630-40aef37c3d50-c000.snappy.parquet
    part-00006-d273f657-9c1f-4dd1-8bbf-fb66eba644f3-c000.snappy.parquet
    part-00007-91fbd325-4e1c-437f-a7c4-e0fd8e91b26d-c000.snappy.parquet
    part-00008-d6982b42-0653-4ef5-8b00-72f3bb62b7ce-c000.snappy.parquet
    part-00009-4539d215-00c7-4f2b-9fc7-cf8c7544070c-c000.snappy.parquet
    part-00010-4238061e-7d3c-43d5-9d29-9b4291b38d55-c000.snappy.parquet
    part-00011-00828917-003b-4eff-a175-81b3e86890cb-c000.snappy.parquet


This folder called the `_delta_log` is the single source of truth for the delta table, and contains all history for a given table; currently there is a single `.json` file, since only one operation was done to this table. This folder is automatically created when a table is created, and is updated for every operation on the delta table.


```python
%ls /storage/data/airline_2m.delta/_delta_log
```

    00000000000000000000.json



```python
(jdf := spark.read.json("/storage/data/airline_2m.delta/_delta_log/00000000000000000000.json")).show()
```

    +--------------------+--------------------+--------------------+--------+
    |                 add|          commitInfo|            metaData|protocol|
    +--------------------+--------------------+--------------------+--------+
    |                null|{Apache-Spark/3.3...|                null|    null|
    |                null|                null|                null|  {1, 2}|
    |                null|                null|{1687486190991, {...|    null|
    |{true, 1687486192...|                null|                null|    null|
    |{true, 1687486192...|                null|                null|    null|
    |{true, 1687486192...|                null|                null|    null|
    |{true, 1687486192...|                null|                null|    null|
    |{true, 1687486192...|                null|                null|    null|
    |{true, 1687486192...|                null|                null|    null|
    |{true, 1687486192...|                null|                null|    null|
    |{true, 1687486192...|                null|                null|    null|
    |{true, 1687486192...|                null|                null|    null|
    |{true, 1687486192...|                null|                null|    null|
    |{true, 1687486192...|                null|                null|    null|
    |{true, 1687486192...|                null|                null|    null|
    +--------------------+--------------------+--------------------+--------+
    


The above is a bit hard to see, to let's filter for the one entry where `commitInfo` is not null. In the commit info, there are several important parameters, namely the ovewrite mode, the operation (in this case `WRITE`) and the timestamp.


```python
from pyspark.sql import functions as F

jdf.filter(F.col('commitInfo').isNotNull()).select('commitInfo').show(truncate=False)
```

    +--------------------------------------------------------------------------------------------------------------------------------------------------------+
    |commitInfo                                                                                                                                              |
    +--------------------------------------------------------------------------------------------------------------------------------------------------------+
    |{Apache-Spark/3.3.2 Delta-Lake/2.3.0, false, Serializable, WRITE, {12, 33238, 75}, {Overwrite, []}, 1687486192334, 0e2eefc4-557d-45b2-ac9f-e1f56b484fbc}|
    +--------------------------------------------------------------------------------------------------------------------------------------------------------+
    


The metadata stores information on the columns, type of columns, constraints on the columns and the type of file (parquet).


```python
jdf.filter(F.col('metaData').isNotNull()).select('metaData').show(truncate=False)
```

    +----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |metaData                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
    +----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |{1687486190991, {parquet}, e0aa9322-351f-40a6-9327-5d72059b0a0c, [], {"type":"struct","fields":[{"name":"FlightDate","type":"timestamp","nullable":true,"metadata":{}},{"name":"Reporting_Airline","type":"string","nullable":true,"metadata":{}},{"name":"Flight_Number_Reporting_Airline","type":"integer","nullable":true,"metadata":{}},{"name":"Origin","type":"string","nullable":true,"metadata":{}},{"name":"Dest","type":"string","nullable":true,"metadata":{}},{"name":"DepTime","type":"integer","nullable":true,"metadata":{}},{"name":"DepDelay","type":"double","nullable":true,"metadata":{}},{"name":"ArrTime","type":"integer","nullable":true,"metadata":{}},{"name":"ArrDelay","type":"double","nullable":true,"metadata":{}}]}}|
    +----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    


A more concise way of viewing this is by calling `.history().show()` on the delta table. Here we see metadata:
- Version of the table
- Timestamp of the given operation
- UserID, Name (person/entity that performed the operation)
- Operation Parameters: write mode, other config operations
- Notebook/ClusterID: these are databricks specific and would not be populated here
- Isolation Level: this controls write conflicts, and the degree to which a transaction must be isolated from modifications made by concurrent operations



```python
# load table
delta_table = DeltaTable.forPath(spark, '/storage/data/airline_2m.delta/')
delta_table.history().show()
```

    +-------+--------------------+------+--------+---------+--------------------+----+--------+---------+-----------+--------------+-------------+--------------------+------------+--------------------+
    |version|           timestamp|userId|userName|operation| operationParameters| job|notebook|clusterId|readVersion|isolationLevel|isBlindAppend|    operationMetrics|userMetadata|          engineInfo|
    +-------+--------------------+------+--------+---------+--------------------+----+--------+---------+-----------+--------------+-------------+--------------------+------------+--------------------+
    |      0|2023-06-22 22:09:...|  null|    null|    WRITE|{mode -> Overwrit...|null|    null|     null|       null|  Serializable|        false|{numFiles -> 12, ...|        null|Apache-Spark/3.3....|
    +-------+--------------------+------+--------+---------+--------------------+----+--------+---------+-----------+--------------+-------------+--------------------+------------+--------------------+
    


# Updating a Delta Table

## Adding a Column

If I were to modify the table in any way, such as by adding a new row, this would be recorded in the transaction-log:


```python
delta_table.toDF().count() # before the modification, we have 75 rows
```




    75



Let's add a new column which is a function of `DepDelay` and `ArrDelay`. For example, say we want to observe the relationship between Departure and Arrival delays by menas of a ratio:


```python
from pyspark.sql import functions as F

delta_table = spark.read.format('delta').load('/storage/data/airline_2m.delta/')

delta_table_updated = (
    delta_table
      .withColumn('dep_to_arr_ratio', F.expr('round(DepDelay/ArrDelay, 3)'))
)

delta_table_updated.show() # great, let's write this back to the delta table
```

    +-------------------+-----------------+-------------------------------+------+----+-------+--------+-------+--------+----------------+
    |         FlightDate|Reporting_Airline|Flight_Number_Reporting_Airline|Origin|Dest|DepTime|DepDelay|ArrTime|ArrDelay|dep_to_arr_ratio|
    +-------------------+-----------------+-------------------------------+------+----+-------+--------+-------+--------+----------------+
    |2017-12-17 00:00:00|               DL|                           2634|   JFK| MCO|   1121|    -4.0|   1359|   -32.0|           0.125|
    |2017-12-07 00:00:00|               B6|                            161|   JFK| SMF|   1820|   -10.0|   2201|    -3.0|           3.333|
    |2017-12-11 00:00:00|               DL|                           2498|   JFK| CHS|    749|   -16.0|    948|   -51.0|           0.314|
    |2017-12-21 00:00:00|               DL|                            456|   JFK| SLC|   1040|    -7.0|   1404|   -16.0|           0.438|
    |2017-12-01 00:00:00|               B6|                            415|   JFK| SFO|    921|    -8.0|   1307|    -3.0|           2.667|
    |2017-12-16 00:00:00|               DL|                            496|   JFK| SFO|   1928|     3.0|   2246|   -36.0|          -0.083|
    |2017-12-01 00:00:00|               B6|                           1507|   JFK| IAD|   2138|     3.0|   2310|    17.0|           0.176|
    |2017-12-05 00:00:00|               B6|                            286|   JFK| ROC|    741|    -9.0|    906|    -3.0|             3.0|
    |2017-12-27 00:00:00|               B6|                           1407|   JFK| IAD|    953|     0.0|   1107|    -7.0|             0.0|
    |2017-12-28 00:00:00|               B6|                            711|   JFK| LAS|   1759|     0.0|   2046|   -13.0|             0.0|
    |2017-12-05 00:00:00|               B6|                           1273|   JFK| CHS|    733|    -7.0|    943|    -8.0|           0.875|
    |2017-12-03 00:00:00|               DL|                            424|   JFK| LAX|    820|     5.0|   1131|    -4.0|           -1.25|
    |2017-12-01 00:00:00|               DL|                            451|   JFK| ATL|    557|    -3.0|    807|   -34.0|           0.088|
    |2017-12-25 00:00:00|               B6|                            583|   JFK| MCO|    658|    -2.0|   1031|    31.0|          -0.065|
    |2017-12-31 00:00:00|               VX|                           1411|   JFK| LAX|   1417|    77.0|   1806|    98.0|           0.786|
    |2017-12-07 00:00:00|               B6|                           1273|   JFK| CHS|    736|    -4.0|    956|     5.0|            -0.8|
    |2017-12-27 00:00:00|               B6|                           1013|   JFK| LGB|   1833|    33.0|   2155|    37.0|           0.892|
    |2017-12-15 00:00:00|               DL|                           2793|   JFK| LAS|    842|    12.0|   1112|   -30.0|            -0.4|
    |2017-12-01 00:00:00|               DL|                           2791|   JFK| CHS|   1353|     0.0|   1557|   -23.0|             0.0|
    |2017-12-29 00:00:00|               B6|                            108|   JFK| PWM|   2333|    48.0|     37|    39.0|           1.231|
    +-------------------+-----------------+-------------------------------+------+----+-------+--------+-------+--------+----------------+
    only showing top 20 rows
    



```python
delta_table_updated.write.mode('overwrite').format('delta').save('/storage/data/airline_2m.delta/') # Hmm, that's not right!!
```


    ---------------------------------------------------------------------------

    AnalysisException                         Traceback (most recent call last)

    Cell In[25], line 1
    ----> 1 delta_table_updated.write.mode('overwrite').format('delta').save('/storage/data/airline_2m.delta/')


    File ~/miniconda3/envs/spark_env/lib/python3.8/site-packages/pyspark/sql/readwriter.py:968, in DataFrameWriter.save(self, path, format, mode, partitionBy, **options)
        966     self._jwrite.save()
        967 else:
    --> 968     self._jwrite.save(path)


    File ~/miniconda3/envs/spark_env/lib/python3.8/site-packages/py4j/java_gateway.py:1321, in JavaMember.__call__(self, *args)
       1315 command = proto.CALL_COMMAND_NAME +\
       1316     self.command_header +\
       1317     args_command +\
       1318     proto.END_COMMAND_PART
       1320 answer = self.gateway_client.send_command(command)
    -> 1321 return_value = get_return_value(
       1322     answer, self.gateway_client, self.target_id, self.name)
       1324 for temp_arg in temp_args:
       1325     temp_arg._detach()


    File ~/miniconda3/envs/spark_env/lib/python3.8/site-packages/pyspark/sql/utils.py:196, in capture_sql_exception.<locals>.deco(*a, **kw)
        192 converted = convert_exception(e.java_exception)
        193 if not isinstance(converted, UnknownException):
        194     # Hide where the exception came from that shows a non-Pythonic
        195     # JVM exception message.
    --> 196     raise converted from None
        197 else:
        198     raise


    AnalysisException: A schema mismatch detected when writing to the Delta table (Table ID: e0aa9322-351f-40a6-9327-5d72059b0a0c).
    To enable schema migration using DataFrameWriter or DataStreamWriter, please set:
    '.option("mergeSchema", "true")'.
    For other operations, set the session configuration
    spark.databricks.delta.schema.autoMerge.enabled to "true". See the documentation
    specific to the operation for details.
    
    Table schema:
    root
    -- FlightDate: timestamp (nullable = true)
    -- Reporting_Airline: string (nullable = true)
    -- Flight_Number_Reporting_Airline: integer (nullable = true)
    -- Origin: string (nullable = true)
    -- Dest: string (nullable = true)
    -- DepTime: integer (nullable = true)
    -- DepDelay: double (nullable = true)
    -- ArrTime: integer (nullable = true)
    -- ArrDelay: double (nullable = true)
    
    
    Data schema:
    root
    -- FlightDate: timestamp (nullable = true)
    -- Reporting_Airline: string (nullable = true)
    -- Flight_Number_Reporting_Airline: integer (nullable = true)
    -- Origin: string (nullable = true)
    -- Dest: string (nullable = true)
    -- DepTime: integer (nullable = true)
    -- DepDelay: double (nullable = true)
    -- ArrTime: integer (nullable = true)
    -- ArrDelay: double (nullable = true)
    -- dep_to_arr_ratio: double (nullable = true)
    
             
    To overwrite your schema or change partitioning, please set:
    '.option("overwriteSchema", "true")'.
    
    Note that the schema can't be overwritten when using
    'replaceWhere'.
             


In the above we get a schema mismatch error, this is because the original delta table did not contain our newly engineered column and implicity enforces schema validation. In order to help identify the error, `spark` prints out the before and after schemas of the delta table. This is an incresibly stringent check, and can be used as a gatekeeper of clean, full transformed data that is ready for production. Schema enforcement prevents data "dilution" which can occur when new columns are added so frequently that the original data loses its meaning due to data deluge. 


If you decide that you absolutely needed to add a new column, we use `mode('overWriteSchema', 'true')` in the write statement, also known as *schema evolution* in the Delta table documentation. There are two types of operations supported via schema *evolution*: adding new columns (what we're doing) and changing data types from `Null` to any other type OR upcasting `Byte` to `Short` or `Integer`.

*Note, using the option `overwrite` here marks the original version of the data as "tombstone", which will cause this version to be removed if Delta's `VACUUM` command is run.*  


```python
delta_table_updated.write.mode('overwrite').option('overwriteSchema', 'true').format('delta').save('/storage/data/airline_2m.delta/') 
```

Now if we re-examine the delta-log, there are two files present here:


```python
%ls /storage/data/airline_2m.delta/_delta_log
```

    00000000000000000000.json  00000000000000000001.json


If we view the delta-table history, we can see that a new `version` has been added, with a new timestamp and the type of operation. (Here we only select a few relevant columns for ease-of-viewing). In the `operationParameters` field, we can see where we passed in `mode`, and the in `operationMetrics` field, we can see that the number of parquet files has not changed (we didn't expect this to change) and neither did the number of rows. However, the `numOutputBytes` has changed due to the addition of our new column:


```python
spark.sql('describe history delta.`/storage/data/airline_2m.delta/`').select(*['version', 'timestamp', 'operation', 'operationParameters', 'operationMetrics']).show(truncate=False)
```

    +-------+-----------------------+---------+--------------------------------------+--------------------------------------------------------------+
    |version|timestamp              |operation|operationParameters                   |operationMetrics                                              |
    +-------+-----------------------+---------+--------------------------------------+--------------------------------------------------------------+
    |1      |2023-06-24 09:26:21.574|WRITE    |{mode -> Overwrite, partitionBy -> []}|{numFiles -> 12, numOutputRows -> 75, numOutputBytes -> 37214}|
    |0      |2023-06-22 22:09:52.338|WRITE    |{mode -> Overwrite, partitionBy -> []}|{numFiles -> 12, numOutputRows -> 75, numOutputBytes -> 33238}|
    +-------+-----------------------+---------+--------------------------------------+--------------------------------------------------------------+
    


## Upsert

This is hyper-specific to delta table syntax, where a combination of an `insert` and an `update` is done in one go. Let's say that I have a new table with updates to apply to the delta table:


```python
from datetime import datetime


# create an updates dataframe
updates = [
    (datetime.strptime('2017-12-17', '%Y-%m-%d'), 'DL', 2634, 'JFK', 'MCO', 1125, 0.0, 1359, -32.0, 0.0), # update existing entry to have no departure delay
    (datetime.strptime('2017-12-26', '%Y-%m-%d'), 'DL',1368, 'JFK', 'MIA', 1107, 1.0, 1421, 1.0, 1.0), # new entry that did not exists
]
(updates_table := spark.createDataFrame(updates, schema=delta_table_updated.schema)).show()
```

    +-------------------+-----------------+-------------------------------+------+----+-------+--------+-------+--------+----------------+
    |         FlightDate|Reporting_Airline|Flight_Number_Reporting_Airline|Origin|Dest|DepTime|DepDelay|ArrTime|ArrDelay|dep_to_arr_ratio|
    +-------------------+-----------------+-------------------------------+------+----+-------+--------+-------+--------+----------------+
    |2017-12-17 00:00:00|               DL|                           2634|   JFK| MCO|   1125|     0.0|   1359|   -32.0|             0.0|
    |2017-07-26 00:00:00|               DL|                           1368|   JFK| MIA|   1107|     1.0|   1421|     1.0|             1.0|
    +-------------------+-----------------+-------------------------------+------+----+-------+--------+-------+--------+----------------+
    


This new data has both a row that matches an existing entry in our dataframe, as well as a new row that did not previously exist. 


```python
delta_table_updated.filter('Flight_Number_Reporting_Airline=1368').show()
```

    +----------+-----------------+-------------------------------+------+----+-------+--------+-------+--------+----------------+
    |FlightDate|Reporting_Airline|Flight_Number_Reporting_Airline|Origin|Dest|DepTime|DepDelay|ArrTime|ArrDelay|dep_to_arr_ratio|
    +----------+-----------------+-------------------------------+------+----+-------+--------+-------+--------+----------------+
    +----------+-----------------+-------------------------------+------+----+-------+--------+-------+--------+----------------+
    


This is the syntax for upserting:
```python
table.alias('old_table')
  .merge(
    source=a_new_table.alias('a_new_table'),
    condition= # keys to match on, in this case we match on FlightData, Origin, Dest, Airline and Flight-Number
  )
```
Following this, there are multiple possible clauses, at least one of which is necessary:
- `whenNotMatchedInsert`: adds new rows if no matches are made
- `whenMatchedUpdate`: does not insert new rows, this is closest to an `UPDATE` statement in SQL

So, in order to update matches:
```python
  .whenMatchedUpdate(set = {
    'Column1': F.col('a_new_table.Column1'), # use value from updates for column1
    'Column2': F.col('old_table.Column2'), # retain value from original dataset for column 2 
  })
```

Finally, to insert data where no matches are made:
```python
  .whenNotMatchedInsertAll()
```

There can be any number of `whenMatched` clauses (at most one update and one delete) action. The update in the merge only updates the specified columns, and multiple `whenMatched` statements execute in the order specified. In order to update all the columns, use `whenMatched(...).updateAll()`, where `...` would be your specified key matches. In the `whenNotMatched` case, this can only have the insert operation, where any unspecified column assume null values.

For the full API reference, see [here](https://docs.delta.io/latest/delta-update.html#language-python)

Performing the actual upsert, we specify conditions to match on FlightDate, Origin, Destination, Flight-Number and Airline, updating the columns for departure/arrival time/delay. Finally, there's a statement to insert new rows where they did not match.


```python
(DeltaTable.forPath(spark, '/storage/data/airline_2m.delta/').alias('current_data')
  # specify merge conditions
  .merge(
      source=updates_table.alias('new_data'), 
      condition=F.expr('current_data.FlightDate = new_data.FlightDate and new_data.Origin = current_data.Origin and current_data.Dest = new_data.Dest and  current_data.Reporting_Airline = new_data.Reporting_Airline and current_data.Flight_Number_Reporting_Airline = new_data.Flight_Number_Reporting_Airline'))
  # update data where matched
  .whenMatchedUpdate(set = {
    'DepTime': F.col('new_data.DepTime'),
    'DepDelay': F.col('new_data.DepDelay'),
    'ArrTime': F.col('new_data.ArrTime'),
    'ArrDelay': F.col('new_data.ArrDelay'),
  })
  # insert where not matched
  .whenNotMatchedInsertAll()
  .execute()
 )

```


```python
# we can see that a third entry has been added to the transaction log
%ls /storage/data/airline_2m.delta/_delta_log
```

    00000000000000000000.json  00000000000000000001.json  00000000000000000002.json



```python
history = DeltaTable.forPath(spark, '/storage/data/airline_2m.delta/').history()
history.show() # that's a bit much, let's inspect individually
```

    +-------+--------------------+------+--------+---------+--------------------+----+--------+---------+-----------+--------------+-------------+--------------------+------------+--------------------+
    |version|           timestamp|userId|userName|operation| operationParameters| job|notebook|clusterId|readVersion|isolationLevel|isBlindAppend|    operationMetrics|userMetadata|          engineInfo|
    +-------+--------------------+------+--------+---------+--------------------+----+--------+---------+-----------+--------------+-------------+--------------------+------------+--------------------+
    |      2|2023-06-24 09:59:...|  null|    null|    MERGE|{predicate -> (((...|null|    null|     null|          1|  Serializable|        false|{numTargetRowsCop...|        null|Apache-Spark/3.3....|
    |      1|2023-06-24 09:26:...|  null|    null|    WRITE|{mode -> Overwrit...|null|    null|     null|          0|  Serializable|        false|{numFiles -> 12, ...|        null|Apache-Spark/3.3....|
    |      0|2023-06-22 22:09:...|  null|    null|    WRITE|{mode -> Overwrit...|null|    null|     null|       null|  Serializable|        false|{numFiles -> 12, ...|        null|Apache-Spark/3.3....|
    +-------+--------------------+------+--------+---------+--------------------+----+--------+---------+-----------+--------------+-------------+--------------------+------------+--------------------+
    



```python
history.select(*['timestamp', 'version','operation']).show() # there is now a third version (number 2), with an operation MERGE
```

    +--------------------+-------+---------+
    |           timestamp|version|operation|
    +--------------------+-------+---------+
    |2023-06-24 09:59:...|      2|    MERGE|
    |2023-06-24 09:26:...|      1|    WRITE|
    |2023-06-22 22:09:...|      0|    WRITE|
    +--------------------+-------+---------+
    


`operationParameters` shows us the substance of the operation, where the `predicate` outlines our merge conditions and the defined action (in this case an `update`), followed by a predicate for non-matched rows:


```python
history.filter('version=2').select('operationParameters').show(truncate=False) 
```

    +-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |operationParameters                                                                                                                                                                                                                                                                                                                                                                                                                                              |
    +-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |{predicate -> ((((current_data.FlightDate = new_data.FlightDate) AND (new_data.Origin = current_data.Origin)) AND (current_data.Dest = new_data.Dest)) AND ((current_data.Reporting_Airline = new_data.Reporting_Airline) AND (current_data.Flight_Number_Reporting_Airline = new_data.Flight_Number_Reporting_Airline))), matchedPredicates -> [{"actionType":"update"}], notMatchedPredicates -> [{"actionType":"insert"}], notMatchedBySourcePredicates -> []}|
    +-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    


Finally, if we view operation metrics, we get a readout for the number of rows added/deleted, files added/deleted, the number of input/output rows etc etc. This is the most useful for logging operations performed on a delta table, and can be used to easily catch any spurios operations, such as the addition of extra rows, schema implosion and file cloning.


```python
history.filter('version=2').select('operationMetrics').show(truncate=False)
```

    +-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |operationMetrics                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
    +-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |{numTargetRowsCopied -> 7, numTargetRowsDeleted -> 0, numTargetFilesAdded -> 1, numTargetBytesAdded -> 3209, numTargetBytesRemoved -> 3218, numTargetRowsMatchedUpdated -> 1, executionTimeMs -> 875, numTargetRowsInserted -> 1, numTargetRowsMatchedDeleted -> 0, scanTimeMs -> 515, numTargetRowsUpdated -> 1, numOutputRows -> 9, numTargetRowsNotMatchedBySourceUpdated -> 0, numTargetChangeFilesAdded -> 0, numSourceRows -> 2, numTargetFilesRemoved -> 1, numTargetRowsNotMatchedBySourceDeleted -> 0, rewriteTimeMs -> 340}|
    +-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    


# Time-Travel

Now, possibly the most useful feature of having a transaction-log is "time-travel". This is essentially being able to access versions of the delta at different points in its lifetime. Let's re-examine the history:


```python
history.select(*['version', 'timestamp']).show()
```

    +-------+--------------------+
    |version|           timestamp|
    +-------+--------------------+
    |      2|2023-06-24 09:59:...|
    |      1|2023-06-24 09:26:...|
    |      0|2023-06-22 22:09:...|
    +-------+--------------------+
    


We can see 3 versions, the original, and two operations. In order to access these previous versions, we need to use one of the (many)  ways of accessing previous revisions:


```python
# reading version 0, we can see that the engineered column added in version 1 does not exist as yet
(spark.read.format('delta').option('versionAsOf', 0).load('/storage/data/airline_2m.delta/')).printSchema()
```

    root
     |-- FlightDate: timestamp (nullable = true)
     |-- Reporting_Airline: string (nullable = true)
     |-- Flight_Number_Reporting_Airline: integer (nullable = true)
     |-- Origin: string (nullable = true)
     |-- Dest: string (nullable = true)
     |-- DepTime: integer (nullable = true)
     |-- DepDelay: double (nullable = true)
     |-- ArrTime: integer (nullable = true)
     |-- ArrDelay: double (nullable = true)
    


Let's read version 1 using slightly different syntax:


```python
history.select(*['version', 'timestamp']).show(truncate=False)
```

    +-------+-----------------------+
    |version|timestamp              |
    +-------+-----------------------+
    |2      |2023-06-24 09:59:29.027|
    |1      |2023-06-24 09:26:21.574|
    |0      |2023-06-22 22:09:52.338|
    +-------+-----------------------+
    


In this version, our new column has been added:


```python
(spark.read.format('delta').option('timestampAsOf', '2023-06-24 09:26:21.574').load('/storage/data/airline_2m.delta/')).printSchema()
```

    root
     |-- FlightDate: timestamp (nullable = true)
     |-- Reporting_Airline: string (nullable = true)
     |-- Flight_Number_Reporting_Airline: integer (nullable = true)
     |-- Origin: string (nullable = true)
     |-- Dest: string (nullable = true)
     |-- DepTime: integer (nullable = true)
     |-- DepDelay: double (nullable = true)
     |-- ArrTime: integer (nullable = true)
     |-- ArrDelay: double (nullable = true)
     |-- dep_to_arr_ratio: double (nullable = true)
    


Finally, if we view version 2, we can see the difference in the number of rows:


```python
(spark.read.format('delta').option('versionAsOf', 1).load('/storage/data/airline_2m.delta/')).count(), (spark.read.format('delta').option('versionAsOf', 2).load('/storage/data/airline_2m.delta/')).count()
```




    (75, 76)



# Vacuum
Time-travel is enabled via the transaction log, however it would be inefficient to store every version of every table forever. As aforementioned, using the `overwrite` option marks previous versions of data for being removed via the `vacuum` command. The other condition for removing files via `vacuum` is files existing after some given time period (by default it's 7 days). If we observe the delta directory now (after three operations):

```bash
  /home ❯ tree /storage/data/airline_2m.delta                                  at  10:43:52
/storage/data/airline_2m.delta
├── _delta_log
│   ├── 00000000000000000000.json
│   ├── 00000000000000000001.json
│   └── 00000000000000000002.json
├── part-00000-716191cb-67cf-4e63-a06d-c3943fb45664-c000.snappy.parquet
├── part-00000-74403ae3-1703-4c27-aa4c-7de9bd1e6fbe-c000.snappy.parquet
├── part-00000-9db93c29-a618-4f69-aa5f-776e1ca1a221-c000.snappy.parquet
├── part-00001-43218537-207b-4569-8d98-7cb1d2959d3d-c000.snappy.parquet
├── part-00001-e1a760f6-c163-4aa5-b327-fd97f40d8509-c000.snappy.parquet
├── part-00002-17bd3789-0052-4a37-aa0a-bd91b4b1fbc7-c000.snappy.parquet
├── part-00002-b41a2670-c5bc-4515-93c6-c9fe87c3d132-c000.snappy.parquet
├── part-00003-0393fe9a-e8cc-4c69-83a4-e11828b75886-c000.snappy.parquet
├── part-00003-bad613ae-6925-4578-9c0e-dfd43d31a8f7-c000.snappy.parquet
├── part-00004-4846b914-1702-446b-861a-52c9e9f080f0-c000.snappy.parquet
├── part-00004-edbda9cf-91b8-4752-bec3-f30e93651fe8-c000.snappy.parquet
├── part-00005-61394241-d08a-4dbc-9f9e-8a2a904c0a18-c000.snappy.parquet
├── part-00005-9c275ad8-871a-4948-9630-40aef37c3d50-c000.snappy.parquet
├── part-00006-1829cf9d-4451-4882-9587-2a92cfdff619-c000.snappy.parquet
├── part-00006-d273f657-9c1f-4dd1-8bbf-fb66eba644f3-c000.snappy.parquet
├── part-00007-91fbd325-4e1c-437f-a7c4-e0fd8e91b26d-c000.snappy.parquet
├── part-00007-aaf31339-c029-4955-80f5-2649b4fe6caf-c000.snappy.parquet
├── part-00008-2340179d-0109-4c50-89ba-f31242beba14-c000.snappy.parquet
├── part-00008-d6982b42-0653-4ef5-8b00-72f3bb62b7ce-c000.snappy.parquet
├── part-00009-4539d215-00c7-4f2b-9fc7-cf8c7544070c-c000.snappy.parquet
├── part-00009-5b136823-d160-403d-b8aa-6fe6126aad2d-c000.snappy.parquet
├── part-00010-4238061e-7d3c-43d5-9d29-9b4291b38d55-c000.snappy.parquet
├── part-00010-c8695154-ff81-43d7-b7c3-f2687192ce7a-c000.snappy.parquet
├── part-00011-00828917-003b-4eff-a175-81b3e86890cb-c000.snappy.parquet
└── part-00011-7fc1ae81-7e4e-4dfd-b8be-fd5a0c43e127-c000.snappy.parquet

1 directory, 28 files
```


```python
# we need to disable some safety checks
spark.conf.set('spark.databricks.delta.retentionDurationCheck.enabled','false')
spark.sql('vacuum airline_2m retain 0 hours').show(truncate=False)
```

                                                                                    

    Deleted 0 files and directories in a total of 1 directories.
    +-------------------------------------------------------------+
    |path                                                         |
    +-------------------------------------------------------------+
    |file:/storage/projects/notes/tools/spark-warehouse/airline_2m|
    +-------------------------------------------------------------+
    


The above would show files removed if there were older files in the delta-history. The safety check is disabled since running `VACUUM` for files which existed less than a week is generally note desirable.
