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
    # Initialize Spark Session with MongoDB connector
    spark = SparkSession.builder \
        .appName("HybridSalesForecastingSystem") \
        .config("spark.mongodb.write.connection.uri", MONGO_URI) \
        .config("spark.mongodb.read.connection.uri", MONGO_URI) \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    # 2. SCHEMA DEFINITION
    # Define the structure of our incoming CSV and Kafka JSON data
    schema = StructType([
        StructField("id", StringType(), True),
        StructField("date", StringType(), True),
        StructField("store_nbr", IntegerType(), True),
        StructField("family", StringType(), True),
        StructField("sales", FloatType(), True),
        StructField("onpromotion", IntegerType(), True)
    ])

    # ---------------------------------------------------------
    # PHASE 1: MODEL TRAINING (Offline Learning)
    # ---------------------------------------------------------
    logger.info("Loading training data...")
    
    try:
        # Load dataset from CSV
        df_raw = spark.read.csv(DATA_FILE, header=True, schema=schema)
        
        # --- CRITICAL STEP 1: DATA CLEANING & FILTERING ---
        # We assume that '0' sales represent closed days or stockouts.
        # Removing them ensures the model learns from active sales days only.
        df_filtered = df_raw.filter("sales > 0")

        # --- CRITICAL STEP 2: RANDOMIZATION & SAMPLING ---
        # Increased training size to 20,000 for better cluster stability and regression accuracy.
        # .orderBy(rand()) ensures we get a random sample, not just the first dates.
        TRAINING_SIZE = 20000 
        df_train = df_filtered.orderBy(rand()).limit(TRAINING_SIZE)
        
        # Verify data availability
        count = df_train.count()
        if count == 0:
            raise Exception("No data found after filtering sales > 0")

        logger.info(f"Training on {count} RANDOMIZED valid records...")

        # --- STEP A: FEATURE ENGINEERING ---
        
        # 1. Product Family Encoding (Categorical -> Numerical Vector)
        indexer_family = StringIndexer(inputCol="family", outputCol="family_idx", handleInvalid="keep")
        encoder_family = OneHotEncoder(inputCols=["family_idx"], outputCols=["family_vec"])
        
        # 2. Store Number Encoding (Location Context)
        indexer_store = StringIndexer(inputCol="store_nbr", outputCol="store_idx", handleInvalid="keep")
        encoder_store = OneHotEncoder(inputCols=["store_idx"], outputCols=["store_vec"])
        
        # --- STEP B: UNSUPERVISED LEARNING (Clustering) ---
        # GOAL: Segment stores based on volume context (Store Type + Product + Promotion).
        # NOTE: Sales are NOT used here to prevent leakage.
        
        assembler_clustering = VectorAssembler(
            inputCols=["store_vec", "family_vec", "onpromotion"], 
            outputCol="features_for_clustering"
        )
        
        # K-Means Algorithm
        # k=3: To find Low, Medium, and High volume segments.
        kmeans = KMeans(featuresCol="features_for_clustering", predictionCol="cluster_id", k=3, seed=42)

        # --- STEP C: SUPERVISED LEARNING (Regression) ---
        # GOAL: Predict exact sales units.
        # Input: All encoded features PLUS the Cluster ID found in Step B.
        
        assembler_final = VectorAssembler(
            inputCols=["store_vec", "family_vec", "onpromotion", "cluster_id"], 
            outputCol="features_final"
        )
        
        # Random Forest Regressor
        # numTrees=20: Balanced for speed and accuracy in streaming context.
        rf = RandomForestRegressor(featuresCol="features_final", labelCol="sales", numTrees=20)

        # --- PIPELINE CONSTRUCTION ---
        # Defines the exact sequence of data transformation and training.
        pipeline = Pipeline(stages=[
            indexer_family, encoder_family,
            indexer_store, encoder_store,
            assembler_clustering, kmeans, 
            assembler_final, rf
        ])

        logger.info("Training Optimized Hybrid Model...")
        model = pipeline.fit(df_train)
        logger.info("Training Completed Successfully.")

    except Exception as e:
        logger.error("Training Error: %s", e)
        return

    # ---------------------------------------------------------
    # PHASE 2: REAL-TIME STREAMING
    # ---------------------------------------------------------
    logger.info("Starting Streaming Phase...")

    # Read from Kafka
    df_stream = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS) \
        .option("subscribe", KAFKA_TOPIC) \
        .option("startingOffsets", "latest") \
        .load()

    # Parse JSON Payload
    df_parsed = df_stream.select(from_json(col("value").cast("string"), schema).alias("data")).select("data.*")
    
    # Handle Nulls for streaming stability
    df_parsed = df_parsed.na.fill(0)

    # Apply the Trained Pipeline Model
    # 1. Transforms features
    # 2. Assigns Cluster ID
    # 3. Predicts Sales
    predictions = model.transform(df_parsed)

    # Select Output Columns
    output_df = predictions.select(
        col("id"),
        col("store_nbr"),
        col("family"),
        col("onpromotion"),
        col("cluster_id"),                 
        col("sales").alias("actual_sales"),          
        col("prediction").alias("predicted_sales")   
    )

    # Writer Function for MongoDB
    def write_to_mongo(df, epoch_id):
        df.write \
            .format("mongodb") \
            .mode("append") \
            .option("spark.mongodb.write.connection.uri", MONGO_URI) \
            .save()

    # Start Streaming Query
    query = output_df.writeStream \
        .outputMode("append") \
        .foreachBatch(write_to_mongo) \
        .option("checkpointLocation", "/tmp/checkpoint") \
        .start()

    query.awaitTermination()

if __name__ == "__main__":
    main()