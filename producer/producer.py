import time
import json
import pandas as pd
from kafka import KafkaProducer

# Configuration
KAFKA_BROKER = 'kafka:9092'
TOPIC = 'store-sales'
DATA_FILE = '/app/dataset/train.csv'

def json_serializer(data):
    return json.dumps(data).encode('utf-8')

def main():
    print(f"Connecting to Kafka at {KAFKA_BROKER}...")
    producer = None
    # Retry mechanism for Kafka connection
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
        # We read the CSV into a pandas DataFrame.
        # Ensure we have enough memory. If file is huge (>2GB), use chunking with random skip.
        # For this project, we assume it fits in memory.
        df = pd.read_csv(DATA_FILE)
        
        # 2. SHUFFLE DATA (CRITICAL STEP)
        # frac=1 means return all rows, but in random order.
        # reset_index drops the old sorted index.
        print("Shuffling dataset to simulate random real-time events...")
        df_shuffled = df.sample(frac=1).reset_index(drop=True)
        
        print(f"Starting stream of {len(df_shuffled)} records...")

        # 3. STREAM DATA
        # We iterate through the shuffled dataframe
        batch_size = 50 # Send in small batches for smoother visualization
        
        for i in range(0, len(df_shuffled), batch_size):
            chunk = df_shuffled.iloc[i:i+batch_size]
            
            for index, row in chunk.iterrows():
                message = row.to_dict()
                producer.send(TOPIC, message)
            
            producer.flush()
            print(f"Sent batch {i} to {i+batch_size}. Sleeping...")
            
            # Sleep to simulate real-time traffic (adjust as needed)
            time.sleep(1.5) 
            
    except FileNotFoundError:
        print(f"Error: File {DATA_FILE} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if producer:
            producer.close()

if __name__ == "__main__":
    main()