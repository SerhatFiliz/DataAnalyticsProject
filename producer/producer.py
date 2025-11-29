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
    for i in range(10):
        try:
            producer = KafkaProducer(
                bootstrap_servers=[KAFKA_BROKER],
                value_serializer=json_serializer
            )
            print("Connected to Kafka!")
            break
        except Exception as e:
            print(f"Connection failed (attempt {i+1}/10): {e}")
            time.sleep(5)

    if not producer:
        print("Could not connect to Kafka. Exiting.")
        return

    print(f"Reading data from {DATA_FILE}...")
    try:
        # Read CSV in chunks to simulate streaming
        chunk_size = 100
        for chunk in pd.read_csv(DATA_FILE, chunksize=chunk_size):
            for index, row in chunk.iterrows():
                message = row.to_dict()
                producer.send(TOPIC, message)
                # print(f"Sent: {message}") # Uncomment for verbose logging
            
            producer.flush()
            print(f"Sent {len(chunk)} records. Sleeping for 1 second...")
            time.sleep(1) # Simulate real-time delay
            
    except FileNotFoundError:
        print(f"Error: File {DATA_FILE} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if producer:
            producer.close()

if __name__ == "__main__":
    main()
