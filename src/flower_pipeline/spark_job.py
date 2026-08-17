# Databricks/Spark portability example.
# Run with pyspark in a Databricks job.

from pyspark.sql import functions as F

df = spark.read.parquet("/mnt/bronze/energy_observations/")
silver = (
    df.dropDuplicates(["asset_id","timestamp"])
      .withColumn("asset_type",F.split("asset_id","_").getItem(0))
      .withColumn("event_date",F.to_date("timestamp"))
)
silver.write.format("delta").mode("append").partitionBy("event_date").saveAsTable(
    "silver.energy_observations"
)
