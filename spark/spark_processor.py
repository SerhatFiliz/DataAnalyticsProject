from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, rand, month, dayofweek, to_date
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.clustering import KMeans
import logging

# --- 1. LOGGING SETUP ---
# Helps us track the training progress in Kubernetes logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 2. CONFIGURATION ---
KAFKA_BOOTSTRAP_SERVERS = "kafka:9092"
KAFKA_TOPIC = "store-sales"
# MongoDB connection URI (Service name: mongodb, Port: 27017, DB: sales_db, Coll: predictions)
MONGO_URI = "mongodb://mongodb:27017/sales_db.predictions"
DATA_FILE = "/app/dataset/train.csv"

def main():
    # --- 3. INITIALIZATION ---
    # Create Spark Session with MongoDB Connector support
    spark = SparkSession.builder \
        .appName("HybridSalesForecastingSystem") \
        .config("spark.mongodb.write.connection.uri", MONGO_URI) \
        .config("spark.mongodb.read.connection.uri", MONGO_URI) \
        .config("spark.sql.shuffle.partitions", "10") \
        .getOrCreate()

    # Reduce log verbosity to avoid clutter
    spark.sparkContext.setLogLevel("WARN")

    # --- 4. SCHEMA DEFINITION ---
    # Define the strict schema for input data (CSV & JSON)
    schema = StructType([
        StructField("id", StringType(), True),
        StructField("date", StringType(), True),
        StructField("store_nbr", IntegerType(), True),
        StructField("family", StringType(), True),
        StructField("sales", FloatType(), True),
        StructField("onpromotion", IntegerType(), True)
    ])

    # --- HELPER: FEATURE ENGINEERING (DATE) ---
    # Extracts useful features from the raw date string.
    # Used in both Training and Streaming phases.
    def add_date_features(df):
        # 1. Convert string to DateType
        df = df.withColumn("parsed_date", to_date(col("date"), "yyyy-MM-dd"))
        # 2. Extract Month (1-12) -> Captures Seasonality
        df = df.withColumn("month", month(col("parsed_date")))
        # 3. Extract Day of Week (1=Mon, 7=Sun) -> Captures Weekly Cycles
        df = df.withColumn("day_of_week", dayofweek(col("parsed_date")))
        # 4. Fill nulls to prevent errors
        return df.fillna(0)

    # ---------------------------------------------------------
    # PHASE 1: MODEL TRAINING (Offline Learning)
    # ---------------------------------------------------------
    logger.info("Loading full training dataset...")
    
    try:
        # Load raw data from CSV
        df_raw = spark.read.csv(DATA_FILE, header=True, schema=schema)
        
        # Filter: Train only on active sales days (Avoids learning '0' for everything)
        df_filtered = df_raw.filter("sales > 0")
        
        # Enrich: Add Date Features
        df_enhanced = add_date_features(df_filtered)

        # Sampling: Use 300,000 records for robust 'Big Data' training
        # .orderBy(rand()) ensures a random distribution from the entire file
        TRAINING_SIZE = 300000 
        logger.info(f"Sampling {TRAINING_SIZE} records for robust training...")
        
        df_train = df_enhanced.orderBy(rand()).limit(TRAINING_SIZE)
        
        # Cache data in memory for faster iteration
        df_train.cache()

        logger.info(f"Training set prepared with {df_train.count()} records.")

        # --- STEP A: PREPROCESSING (Encoders) ---
        # Convert Categorical strings to Numerical indices/vectors
        indexer_family = StringIndexer(inputCol="family", outputCol="family_idx", handleInvalid="keep")
        encoder_family = OneHotEncoder(inputCols=["family_idx"], outputCols=["family_vec"])
        
        indexer_store = StringIndexer(inputCol="store_nbr", outputCol="store_idx", handleInvalid="keep")
        encoder_store = OneHotEncoder(inputCols=["store_idx"], outputCols=["store_vec"])
        
        # --- STEP B: CLUSTERING (K-Means) ---
        # Goal: Segment data based on context (Store + Product + Time + Promo)
        # We do NOT use 'sales' here to prevent data leakage.
        assembler_clustering = VectorAssembler(
            inputCols=["store_vec", "family_vec", "onpromotion", "month", "day_of_week"], 
            outputCol="features_for_clustering"
        )
        
        kmeans = KMeans(featuresCol="features_for_clustering", predictionCol="cluster_id", k=3, seed=42)

        # --- STEP C: REGRESSION (Random Forest) ---
        # Goal: Predict Sales using all features + the Cluster ID
        assembler_final = VectorAssembler(
            inputCols=["store_vec", "family_vec", "onpromotion", "cluster_id", "month", "day_of_week"], 
            outputCol="features_final"
        )
        
        rf = RandomForestRegressor(featuresCol="features_final", labelCol="sales", numTrees=25, maxDepth=10)

        # --- PIPELINE ---
        # Chaining all steps together
        pipeline = Pipeline(stages=[
            indexer_family, encoder_family,
            indexer_store, encoder_store,
            assembler_clustering, kmeans, 
            assembler_final, rf
        ])

        logger.info("Starting Hybrid Model Training (This might take 1-2 minutes)...")
        model = pipeline.fit(df_train)
        logger.info("Training Completed Successfully.")
        
        # Free up memory
        df_train.unpersist()

    except Exception as e:
        logger.error("Critical Training Error: %s", e)
        return

    # ---------------------------------------------------------
    # PHASE 2: REAL-TIME STREAMING
    # ---------------------------------------------------------
    logger.info("Initializing Kafka Stream...")

    # Read from Kafka Topic
    df_stream = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS) \
        .option("subscribe", KAFKA_TOPIC) \
        .option("startingOffsets", "latest") \
        .load()

    # Parse JSON Payload
    df_parsed = df_stream.select(from_json(col("value").cast("string"), schema).alias("data")).select("data.*")
    
    # Enrich Stream with Date Features (Real-time Feature Engineering)
    df_stream_enhanced = add_date_features(df_parsed)

    # Make Predictions using the trained model
    predictions = model.transform(df_stream_enhanced)

    # Select Final Output Columns for MongoDB
    output_df = predictions.select(
        col("id"),
        col("store_nbr"),
        col("family"),
        col("onpromotion"),
        col("month"),          
        col("day_of_week"),    
        col("cluster_id"),                 
        col("sales").alias("actual_sales"),          
        col("prediction").alias("predicted_sales")   
    )

    # Write Stream to MongoDB
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