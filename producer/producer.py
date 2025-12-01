import time
import json
import pandas as pd
from kafka import KafkaProducer

# Configuration variables for the Kafka Producer.
KAFKA_BROKER = 'kafka:9092'  # Internal Kubernetes service name for the Kafka broker.
TOPIC = 'store-sales'        # The target topic for data streaming.
DATA_FILE = '/app/dataset/train.csv' # Path to the raw input dataset within the container.

def json_serializer(data):
    # Function to serialize Python dictionary data into a UTF-8 encoded JSON payload.
    return json.dumps(data).encode('utf-8')

def main():
    print(f"Connecting to Kafka at {KAFKA_BROKER}...")
    producer = None
    # Implement a 15-attempt retry loop for robust initial Kafka broker connection.
    for i in range(15):
        try:
            producer = KafkaProducer(
                bootstrap_servers=[KAFKA_BROKER],
                value_serializer=json_serializer
            )
            print("Connected to Kafka!")
            break
        except Exception as e:
            print(f"Connection failed (attempt {i+1}/15): {e}")
            time.sleep(5)

    if not producer:
        print("Could not connect to Kafka. Exiting.")
        return

    print(f"Reading and Shuffling data from {DATA_FILE}...")
    try:
        # 1. READ ALL DATA
        # Load the CSV file into a Pandas DataFrame for in-memory processing.
        df = pd.read_csv(DATA_FILE)
        
        # 2. SHUFFLE DATA (CRITICAL STEP)
        # Randomly sample the dataset to simulate asynchronous, real-time transaction arrivals.
        print("Shuffling dataset to simulate random real-time events...")
        df_shuffled = df.sample(frac=1).reset_index(drop=True)
        
        print(f"Starting stream of {len(df_shuffled)} records...")

        # 3. STREAM DATA
        # Iterate through the shuffled data, processing it in defined batches.
        batch_size = 50 # Define the size of the micro-batch sent to Kafka.
        
        for i in range(0, len(df_shuffled), batch_size):
            chunk = df_shuffled.iloc[i:i+batch_size]
            
            for index, row in chunk.iterrows():
                # Convert the DataFrame row to a dictionary and send it as a message.
                message = row.to_dict()
                producer.send(TOPIC, message)
            
            # Explicitly flush the producer to ensure immediate message delivery.
            producer.flush()
            print(f"Sent batch {i} to {i+batch_size}. Sleeping...")
            
            # Introduce a time delay to simulate real-world data latency and flow control.
            time.sleep(1.5) 
            
    except FileNotFoundError:
        print(f"Error: File {DATA_FILE} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Ensure the Kafka producer client connection is properly closed.
        if producer:
            producer.close()

if __name__ == "__main__":
    main()