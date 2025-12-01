# ==============================================================================
# SCRIPT: deploy.ps1
# PROJECT: Hybrid Sales Analytics System
# DESCRIPTION: Automated deployment script for cleaning up existing resources, 
# building Docker images, and deploying the entire infrastructure 
# (Kafka, Mongo, Spark, Dashboard) to Kubernetes.
# ==============================================================================

Write-Host ">>> STARTING BIG DATA ANALYTICS PROJECT DEPLOYMENT <<<" -ForegroundColor Green

# 1. Clean Up Previous Deployment (Clean Slate)
Write-Host "1/5 Cleaning up previous deployments..." -ForegroundColor Yellow
# Delete all resources defined in the k8s folder (Services, Deployments, etc.).
kubectl delete -f k8s/ --ignore-not-found=true
# Explicitly delete lingering Job resources (Producer and Spark).
kubectl delete job producer-job --ignore-not-found=true
kubectl delete job spark-job --ignore-not-found=true

# 2. Build Docker Images
Write-Host "2/5 Building Docker Images..." -ForegroundColor Cyan
# Build the Producer container image.
docker build -t producer:latest ./producer
# Build the Raw Consumer container image.
docker build -t raw-consumer:latest ./raw_consumer
# Build the Spark application (Driver/Executor) image.
docker build -t spark-app:latest ./spark
# Build the Streamlit Dashboard image.
docker build -t dashboard-app:latest ./dashboard

# 3. Deploy Core Infrastructure (DB, Kafka, Zookeeper, Dashboard)
Write-Host "3/5 Deploying Core Infrastructure (Kafka, Mongo, Dashboard)..." -ForegroundColor Cyan
# Deploy Zookeeper (Kafka prerequisite).
kubectl apply -f k8s/zookeeper.yaml
# Deploy Kafka Broker.
kubectl apply -f k8s/kafka.yaml
# Deploy MongoDB with Persistent Volume Claim.
kubectl apply -f k8s/mongodb.yaml
# Deploy Spark Role-Based Access Control (for dynamic executor spawning).
kubectl apply -f k8s/spark-rbac.yaml
# Deploy the Streamlit Dashboard (Presentation Layer).
kubectl apply -f k8s/dashboard.yaml

Write-Host "Waiting 45 seconds for infrastructure services (DB/Kafka) to initialize..." -ForegroundColor Magenta
Start-Sleep -Seconds 45

# 4. Start Consumers (Listener Applications)
Write-Host "4/5 Starting Consumers (Raw Data & Spark ML Job)..." -ForegroundColor Cyan
# Deploy the Raw Consumer to save stream data to MongoDB.
kubectl apply -f k8s/raw-consumer.yaml
# Deploy the Spark ML Job (Driver) to start the Structured Streaming pipeline.
kubectl apply -f k8s/spark-job.yaml

Write-Host "Waiting 20 seconds for the Spark Job to allocate resources and start processing..." -ForegroundColor Magenta
Start-Sleep -Seconds 20

# 5. Start Producer (Data Flow Initialization)
Write-Host "5/5 Starting Producer (Data Source) to begin data flow..." -ForegroundColor Cyan
# Deploy the Producer Job to stream data from CSV to Kafka.
kubectl apply -f k8s/producer-job.yaml

Write-Host "------------------------------------------------" -ForegroundColor Green
Write-Host "DEPLOYMENT COMPLETE!" -ForegroundColor Green
Write-Host "Access the Dashboard at:" -ForegroundColor White
Write-Host "http://localhost:30005" -ForegroundColor Yellow
Write-Host "------------------------------------------------" -ForegroundColor Green