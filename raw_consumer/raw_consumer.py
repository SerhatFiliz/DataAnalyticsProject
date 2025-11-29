import json
import os
import time
from kafka import KafkaConsumer
from pymongo import MongoClient

KAFKA_BROKER = 'kafka:9092'
TOPIC = 'store-sales'
MONGO_URI = 'mongodb://mongodb:27017/'

def main():
    # Mongo Bağlantısı
    client = MongoClient(MONGO_URI)
    db = client['sales_db']
    collection = db['raw_data']

    print("Raw Consumer baslatiliyor...")
    consumer = None
    while not consumer:
        try:
            consumer = KafkaConsumer(
                TOPIC,
                bootstrap_servers=[KAFKA_BROKER],
                auto_offset_reset='latest',
                value_deserializer=lambda x: json.loads(x.decode('utf-8'))
            )
            print("Kafka bağlantısı başarılı!")
        except:
            print("Kafka bekleniyor...")
            time.sleep(5)

    for message in consumer:
        data = message.value
        collection.insert_one(data)
        print(f"Ham veri kaydedildi: {data.get('id')}")

if __name__ == "__main__":
    main()