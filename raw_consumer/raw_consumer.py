import json
import os
import time
from kafka import KafkaConsumer
from pymongo import MongoClient

# Define configuration constants referencing Kubernetes Service names.
KAFKA_BROKER = 'kafka:9092'     # Internal address of the Kafka broker service.
TOPIC = 'store-sales'           # The topic carrying raw transaction data.
MONGO_URI = 'mongodb://mongodb:27017/' # Internal address of the MongoDB service.

def main():
    # Establish MongoDB connection for raw data persistence.
    client = MongoClient(MONGO_URI)
    db = client['sales_db']
    # Select the 'raw_data' collection used for the EDA dashboard tab.
    collection = db['raw_data']

    print("Raw Consumer is starting...")
    consumer = None
    # Connection loop to handle Kafka broker startup dependencies.
    while not consumer:
        try:
            consumer = KafkaConsumer(
                TOPIC,
                bootstrap_servers=[KAFKA_BROKER],
                # Start consuming from the latest offset if no initial offset is found.
                auto_offset_reset='latest',
                # Define deserializer to convert byte array Kafka messages to JSON objects.
                value_deserializer=lambda x: json.loads(x.decode('utf-8'))
            )
            print("Kafka connection successful!")
        except:
            print("Waiting for Kafka...")
            time.sleep(5)

    # Core processing loop: continuously consume messages from the topic.
    for message in consumer:
        data = message.value
        # Persist the raw, unprocessed transaction data into MongoDB.
        collection.insert_one(data)
        print(f"Raw data saved: {data.get('id')}")

if __name__ == "__main__":
    main()