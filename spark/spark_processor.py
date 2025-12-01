from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, rand, month, dayofweek, to_date
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.clustering import KMeans
import logging

# --- 1. LOGGING SETUP ---
# Initialize standard logging for tracking the application status in Kubernetes logs.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 2. CONFIGURATION ---
KAFKA_BOOTSTRAP_SERVERS = "kafka:9092" # Define the internal Kafka service connection endpoint.
KAFKA_TOPIC = "store-sales" # Specify the input topic for streaming data ingestion.
# Define the output URI for prediction results persistence in MongoDB.
MONGO_URI = "mongodb://mongodb:27017/sales_db.predictions" 
DATA_FILE = "/app/dataset/train.csv" # Specify the path to the static training dataset.

def main():
    # --- 3. INITIALIZATION ---
    # Create Spark Session with MongoDB Connector packages for read/write operations.
    spark = SparkSession.builder \
        .appName("HybridSalesForecastingSystem") \
        .config("spark.mongodb.write.connection.uri", MONGO_URI) \
        .config("spark.mongodb.read.connection.uri", MONGO_URI) \
        .config("spark.sql.shuffle.partitions", "10") \
        .getOrCreate()

    # Reduce log verbosity to avoid cluttering Kubernetes logs during execution.
    spark.sparkContext.setLogLevel("WARN")

    # --- 4. SCHEMA DEFINITION ---
    # Define the strict schema for input data validation from both CSV and Kafka JSON streams.
    schema = StructType([
        StructField("id", StringType(), True),
        StructField("date", StringType(), True),
        StructField("store_nbr", IntegerType(), True),
        StructField("family", StringType(), True),
        StructField("sales", FloatType(), True),
        StructField("onpromotion", IntegerType(), True)
    ])

    # --- HELPER: FEATURE ENGINEERING (DATE) ---
    # Function to extract temporal features (Month, Day of Week) from the date column.
    def add_date_features(df):
        # Convert the raw string date field into a DateType column.
        df = df.withColumn("parsed_date", to_date(col("date"), "yyyy-MM-dd"))
        # Extract the Month (1-12) to capture seasonal variance.
        df = df.withColumn("month", month(col("parsed_date")))
        # Extract the Day of Week (1=Mon, 7=Sun) to capture weekly periodicity.
        df = df.withColumn("day_of_week", dayofweek(col("parsed_date")))
        # Fill any generated null values with zero to maintain data integrity.
        return df.fillna(0)

    # ---------------------------------------------------------
    # PHASE 1: MODEL TRAINING (Offline Learning)
    # ---------------------------------------------------------
    logger.info("Loading full training dataset...")
    
    try:
        # Load raw data from the HostPath mounted CSV file.
        df_raw = spark.read.csv(DATA_FILE, header=True, schema=schema)
        
        # Filter out records where sales are zero to focus the model on active transactions.
        df_filtered = df_raw.filter("sales > 0")
        
        # Enrich the dataset by applying the feature engineering helper function.
        df_enhanced = add_date_features(df_filtered)

        # Randomly sample a large subset for robust training and manageable memory usage.
        TRAINING_SIZE = 300000 
        logger.info(f"Sampling {TRAINING_SIZE} records for robust training...")
        
        df_train = df_enhanced.orderBy(rand()).limit(TRAINING_SIZE)
        
        # Cache the processed training data in memory for faster pipeline fitting.
        df_train.cache()

        logger.info(f"Training set prepared with {df_train.count()} records.")

        # --- STEP A: PREPROCESSING (Encoders) ---
        # Convert categorical fields ('family') into numerical indices.
        indexer_family = StringIndexer(inputCol="family", outputCol="family_idx", handleInvalid="keep")
        # Apply one-hot encoding to prevent ordinal relationship assumptions.
        encoder_family = OneHotEncoder(inputCols=["family_idx"], outputCols=["family_vec"])
        
        # Index the store number.
        indexer_store = StringIndexer(inputCol="store_nbr", outputCol="store_idx", handleInvalid="keep")
        # One-hot encode the store index.
        encoder_store = OneHotEncoder(inputCols=["store_idx"], outputCols=["store_vec"])
        
        # --- STEP B: CLUSTERING (K-Means) ---
        # Assemble features *excluding* 'sales' to avoid label leakage in clustering.
        assembler_clustering = VectorAssembler(
            inputCols=["store_vec", "family_vec", "onpromotion", "month", "day_of_week"], 
            outputCol="features_for_clustering"
        )
        
        # Apply K-Means (k=3) to segment transactions into volume profiles.
        kmeans = KMeans(featuresCol="features_for_clustering", predictionCol="cluster_id", k=3, seed=42)

        # --- STEP C: REGRESSION (Random Forest) ---
        # Assemble the final feature vector, including the newly generated 'cluster_id'.
        assembler_final = VectorAssembler(
            inputCols=["store_vec", "family_vec", "onpromotion", "cluster_id", "month", "day_of_week"], 
            outputCol="features_final"
        )
        
        # Initialize the Random Forest Regressor for non-linear sales prediction.
        rf = RandomForestRegressor(featuresCol="features_final", labelCol="sales", numTrees=25, maxDepth=10)

        # --- PIPELINE ---
        # Chain all preprocessing, clustering, and regression stages into a single model pipeline.
        pipeline = Pipeline(stages=[
            indexer_family, encoder_family,
            indexer_store, encoder_store,
            assembler_clustering, kmeans, 
            assembler_final, rf
        ])

        logger.info("Starting Hybrid Model Training (This might take 1-2 minutes)...")
        # Execute the training pipeline on the prepared dataset.
        model = pipeline.fit(df_train)
        logger.info("Training Completed Successfully.")
        
        # Release the cached training data to free up executor memory.
        df_train.unpersist()

    except Exception as e:
        logger.error("Critical Training Error: %s", e)
        # Terminate if training fails as the streaming phase cannot proceed without a model.
        return

    # ---------------------------------------------------------
    # PHASE 2: REAL-TIME STREAMING
    # ---------------------------------------------------------
    logger.info("Initializing Kafka Stream...")

    # Read from Kafka Topic
    # Configure Spark Structured Streaming to consume micro-batches from the specified topic.
    df_stream = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS) \
        .option("subscribe", KAFKA_TOPIC) \
        .option("startingOffsets", "latest") \
        .load()

    # Parse JSON Payload
    # Extract the payload value and apply the defined schema for structured access.
    df_parsed = df_stream.select(from_json(col("value").cast("string"), schema).alias("data")).select("data.*")
    
    # Enrich Stream with Date Features (Real-time Feature Engineering)
    # Apply the same feature engineering logic used during the training phase to the stream.
    df_stream_enhanced = add_date_features(df_parsed)

    # Make Predictions using the trained model
    # Apply the complete fitted ML pipeline model to the enhanced stream data.
    predictions = model.transform(df_stream_enhanced)

    # Select Final Output Columns for MongoDB
    # Rename columns for clarity and select only relevant prediction fields for persistence.
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

    # Write Stream to MongoDB (ForeachBatch Sink)
    # Define a custom function to write each micro-batch to the MongoDB sink.
    def write_to_mongo(df, epoch_id):
        df.write \
            .format("mongodb") \
            .mode("append") \
            .option("spark.mongodb.write.connection.uri", MONGO_URI) \
            .save()

    # Start the Structured Streaming Query.
    query = output_df.writeStream \
        .outputMode("append") \
        .foreachBatch(write_to_mongo) \
        .option("checkpointLocation", "/tmp/checkpoint") \
        .start()

    # Block the thread until the query terminates, keeping the Spark process alive.
    query.awaitTermination()

if __name__ == "__main__":
    main()