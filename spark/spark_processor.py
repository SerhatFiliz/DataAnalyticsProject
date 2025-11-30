from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, rand
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.clustering import KMeans
import logging

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
KAFKA_BOOTSTRAP_SERVERS = "kafka:9092"
KAFKA_TOPIC = "store-sales"
MONGO_URI = "mongodb://mongodb:27017/sales_db.predictions"
DATA_FILE = "/app/dataset/train.csv"

def main():
    # 1. INITIALIZATION
    # Spark Session başlatılıyor. (Yorum satırını zincirin dışına aldım)
    spark = SparkSession.builder \
        .appName("HybridSalesForecastingSystem") \
        .config("spark.mongodb.write.connection.uri", MONGO_URI) \
        .config("spark.mongodb.read.connection.uri", MONGO_URI) \
        .config("spark.sql.shuffle.partitions", "10") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    # 2. SCHEMA DEFINITION
    schema = StructType([
        StructField("id", StringType(), True),
        StructField("date", StringType(), True),
        StructField("store_nbr", IntegerType(), True),
        StructField("family", StringType(), True),
        StructField("sales", FloatType(), True),
        StructField("onpromotion", IntegerType(), True)
    ])

    # ---------------------------------------------------------
    # PHASE 1: MODEL TRAINING (Offline Learning with BIG DATA)
    # ---------------------------------------------------------
    logger.info("Loading full training dataset...")
    
    try:
        # Load the raw CSV file
        df_raw = spark.read.csv(DATA_FILE, header=True, schema=schema)
        
        # --- DATA FILTERING ---
        df_filtered = df_raw.filter("sales > 0")

        # --- DATA SAMPLING (300,000) ---
        TRAINING_SIZE = 300000 
        logger.info(f"Sampling {TRAINING_SIZE} records for robust training...")
        
        # Shuffle and Limit
        df_train = df_filtered.orderBy(rand()).limit(TRAINING_SIZE)
        
        # Cache for performance
        df_train.cache()

        logger.info(f"Training set prepared with {df_train.count()} records.")

        # --- STEP A: FEATURE ENGINEERING ---
        
        # 1. Encode 'family'
        indexer_family = StringIndexer(inputCol="family", outputCol="family_idx", handleInvalid="keep")
        encoder_family = OneHotEncoder(inputCols=["family_idx"], outputCols=["family_vec"])
        
        # 2. Encode 'store_nbr'
        indexer_store = StringIndexer(inputCol="store_nbr", outputCol="store_idx", handleInvalid="keep")
        encoder_store = OneHotEncoder(inputCols=["store_idx"], outputCols=["store_vec"])
        
        # --- STEP B: UNSUPERVISED LEARNING (CLUSTERING) ---
        assembler_clustering = VectorAssembler(
            inputCols=["store_vec", "family_vec", "onpromotion"], 
            outputCol="features_for_clustering"
        )
        
        # K-Means: k=3
        kmeans = KMeans(featuresCol="features_for_clustering", predictionCol="cluster_id", k=3, seed=42)

        # --- STEP C: SUPERVISED LEARNING (REGRESSION) ---
        assembler_final = VectorAssembler(
            inputCols=["store_vec", "family_vec", "onpromotion", "cluster_id"], 
            outputCol="features_final"
        )
        
        # Random Forest Regressor
        rf = RandomForestRegressor(featuresCol="features_final", labelCol="sales", numTrees=25, maxDepth=10)

        # --- PIPELINE ---
        pipeline = Pipeline(stages=[
            indexer_family, encoder_family,
            indexer_store, encoder_store,
            assembler_clustering, kmeans, 
            assembler_final, rf
        ])

        logger.info("Starting Hybrid Model Training (This might take 1-2 minutes)...")
        model = pipeline.fit(df_train)
        logger.info("Training Completed Successfully.")
        
        # Clear memory
        df_train.unpersist()

    except Exception as e:
        logger.error("Critical Training Error: %s", e)
        return

    # ---------------------------------------------------------
    # PHASE 2: REAL-TIME STREAMING
    # ---------------------------------------------------------
    logger.info("Initializing Kafka Stream...")

    df_stream = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS) \
        .option("subscribe", KAFKA_TOPIC) \
        .option("startingOffsets", "latest") \
        .load()

    # Parse Payload
    df_parsed = df_stream.select(from_json(col("value").cast("string"), schema).alias("data")).select("data.*")
    df_parsed = df_parsed.na.fill(0)

    # Inference (Prediction)
    predictions = model.transform(df_parsed)

    # Select Output
    output_df = predictions.select(
        col("id"),
        col("store_nbr"),
        col("family"),
        col("onpromotion"),
        col("cluster_id"),                 
        col("sales").alias("actual_sales"),          
        col("prediction").alias("predicted_sales")   
    )

    # MongoDB Sink
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