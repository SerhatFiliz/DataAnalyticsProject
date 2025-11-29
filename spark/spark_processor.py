from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, struct, to_json
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, FloatType
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
KAFKA_BOOTSTRAP_SERVERS = "kafka:9092"
KAFKA_TOPIC = "store-sales"
MONGO_URI = "mongodb://mongodb:27017/sales_db.predictions"
DATA_FILE = "/app/dataset/train.csv"

def main():
    spark = SparkSession.builder \
        .appName("StoreSalesForecasting") \
        .config("spark.mongodb.write.connection.uri", MONGO_URI) \
        .config("spark.mongodb.read.connection.uri", MONGO_URI) \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    # Define Schema for Store Sales Data
    # id,date,store_nbr,family,sales,onpromotion
    schema = StructType([
        StructField("id", StringType(), True), # ID might be int in CSV but string in JSON safe
        StructField("date", StringType(), True),
        StructField("store_nbr", IntegerType(), True),
        StructField("family", StringType(), True),
        StructField("sales", FloatType(), True),
        StructField("onpromotion", IntegerType(), True)
    ])

    # --- 1. Train Model on Static Data (Startup) ---
    logger.info("Loading training data from %s...", DATA_FILE)
    try:
        # Load a subset for quick training
        df_train = spark.read.csv(DATA_FILE, header=True, schema=schema).limit(10000)
        
        # Fill NA
        df_train = df_train.na.fill(0)

        # Feature Engineering Pipeline
        indexer = StringIndexer(inputCol="family", outputCol="family_index", handleInvalid="keep")
        encoder = OneHotEncoder(inputCols=["family_index"], outputCols=["family_vec"])
        assembler = VectorAssembler(inputCols=["store_nbr", "onpromotion", "family_vec"], outputCol="features")
        rf = RandomForestRegressor(featuresCol="features", labelCol="sales", numTrees=10)

        pipeline = Pipeline(stages=[indexer, encoder, assembler, rf])

        logger.info("Training RandomForest Model...")
        model = pipeline.fit(df_train)
        logger.info("Model training completed.")

    except Exception as e:
        logger.error("Failed to train model: %s", e)
        return

    # --- 2. Stream Processing ---
    logger.info("Starting Stream Processing from Kafka...")

    # Read from Kafka
    df_stream = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS) \
        .option("subscribe", KAFKA_TOPIC) \
        .option("startingOffsets", "latest") \
        .load()

    # Parse JSON
    df_parsed = df_stream.select(from_json(col("value").cast("string"), schema).alias("data")).select("data.*")
    
    # Fill NA for stream
    df_parsed = df_parsed.na.fill(0)

    # Apply Model
    predictions = model.transform(df_parsed)

    # Select output columns
    output_df = predictions.select(
        col("id"),
        col("store_nbr"),
        col("family"),
        col("onpromotion"),
        col("prediction").alias("predicted_sales")
    )

    # Write to MongoDB
    # Write to MongoDB using foreachBatch
    def write_to_mongo(df, epoch_id):
        df.write \
            .format("mongodb") \
            .mode("append") \
            .option("spark.mongodb.write.connection.uri", MONGO_URI) \
            .save()

    query = output_df.writeStream \
        .outputMode("append") \
        .foreachBatch(write_to_mongo) \
        .option("checkpointLocation", "/tmp/checkpoint") \
        .start()

    query.awaitTermination()

if __name__ == "__main__":
    main()
