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
kubectl delete -f k8s/ --ignore-not-found=true
kubectl delete job producer-job --ignore-not-found=true
kubectl delete job spark-job --ignore-not-found=true

# 2. Build Docker Images
Write-Host "2/5 Building Docker Images..." -ForegroundColor Cyan
docker build -t producer:latest ./producer
docker build -t raw-consumer:latest ./raw_consumer
docker build -t spark-app:latest ./spark
docker build -t dashboard-app:latest ./dashboard

# 3. Deploy Core Infrastructure (DB, Kafka, Zookeeper, Dashboard)
Write-Host "3/5 Deploying Core Infrastructure (Kafka, Mongo, Dashboard)..." -ForegroundColor Cyan
kubectl apply -f k8s/zookeeper.yaml
kubectl apply -f k8s/kafka.yaml
kubectl apply -f k8s/mongodb.yaml
kubectl apply -f k8s/spark-rbac.yaml
kubectl apply -f k8s/dashboard.yaml

Write-Host "Waiting 45 seconds for infrastructure services (DB/Kafka) to initialize..." -ForegroundColor Magenta
Start-Sleep -Seconds 45

# 4. Start Consumers (Listener Applications)
Write-Host "4/5 Starting Consumers (Raw Data & Spark ML Job)..." -ForegroundColor Cyan
kubectl apply -f k8s/raw-consumer.yaml
kubectl apply -f k8s/spark-job.yaml

Write-Host "Waiting 20 seconds for the Spark Job to allocate resources and start processing..." -ForegroundColor Magenta
Start-Sleep -Seconds 20

# 5. Start Producer (Data Flow Initialization)
Write-Host "5/5 Starting Producer (Data Source) to begin data flow..." -ForegroundColor Cyan
kubectl apply -f k8s/producer-job.yaml

Write-Host "------------------------------------------------" -ForegroundColor Green
Write-Host "DEPLOYMENT COMPLETE!" -ForegroundColor Green
Write-Host "Access the Dashboard at:" -ForegroundColor White
Write-Host "http://localhost:30005" -ForegroundColor Yellow
Write-Host "------------------------------------------------" -ForegroundColor Green

# ==============================================================================
# PROJECT MANAGEMENT FUNCTIONS (Single-Command Stop/Start)
# ==============================================================================

function Stop-Project {
    Write-Host "🚧 Scaling down Deployments to zero (Stopping resource consumption)..." -ForegroundColor Yellow
    
    # 1. Terminate running Job resources (Spark ML Job and Producer).
    kubectl delete job spark-job producer-job --ignore-not-found=true

    # 2. Scale down all core services to zero replicas.
    kubectl scale deployment --replicas=0 dashboard kafka zookeeper mongodb raw-consumer

    Write-Host "✅ Project successfully paused. Resource usage is now zero." -ForegroundColor Green
}

function Start-Project {
    Write-Host "▶️ Scaling Deployments back to one replica (Resuming services)..." -ForegroundColor Yellow
    
    # 1. Scale up all core services to one replica.
    kubectl scale deployment --replicas=1 dashboard kafka zookeeper mongodb raw-consumer

    Write-Host "⏳ Waiting 30 seconds for the necessary services (Kafka, Mongo) to initialize..." -ForegroundColor Magenta
    Start-Sleep -Seconds 30

    # 2. Restart Spark and Producer (These were deleted in the Stop phase).
    kubectl apply -f k8s/spark-job.yaml
    kubectl apply -f k8s/producer-job.yaml
    
    Write-Host '🔥 Spark and Producer restarted. Please refresh the Dashboard.' -ForegroundColor Green
}