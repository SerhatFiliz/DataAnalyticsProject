# deploy.ps1
Write-Host ">>> DATA ANALYTICS PROJESI KURULUMU BASLIYOR <<<" -ForegroundColor Green

# 1. Eski Kurulumu Temizle (Clean Slate)
Write-Host "Eski deploymentlar temizleniyor..." -ForegroundColor Yellow
kubectl delete -f k8s/ --ignore-not-found=true
# Jobları özel olarak sil
kubectl delete job producer-job --ignore-not-found=true
kubectl delete job spark-job --ignore-not-found=true

# 2. Docker İmajlarını Derle
Write-Host "1/4 Docker Imajları Derleniyor..." -ForegroundColor Cyan
docker build -t producer:latest ./producer
docker build -t raw-consumer:latest ./raw_consumer
docker build -t spark-app:latest ./spark
docker build -t dashboard-app:latest ./dashboard

# 3. Altyapıyı Kur (DB, Kafka, Zookeeper, Dashboard)
Write-Host "2/4 Altyapı (Kafka, Mongo, Dashboard) Kuruluyor..." -ForegroundColor Cyan
kubectl apply -f k8s/zookeeper.yaml
kubectl apply -f k8s/kafka.yaml
kubectl apply -f k8s/mongodb.yaml
kubectl apply -f k8s/spark-rbac.yaml
kubectl apply -f k8s/dashboard.yaml

Write-Host "Altyapının hazir olmasi icin 45 saniye bekleniyor..." -ForegroundColor Magenta
Start-Sleep -Seconds 45

# 4. Consumer'ları Başlat (Dinlemeye başlasınlar)
Write-Host "3/4 Consumerlar (Spark & Raw) Baslatiliyor..." -ForegroundColor Cyan
kubectl apply -f k8s/raw-consumer.yaml
kubectl apply -f k8s/spark-job.yaml

Write-Host "Spark'ın baslamasi icin 20 saniye bekleniyor..." -ForegroundColor Magenta
Start-Sleep -Seconds 20

# 5. Producer'ı Başlat (Veri akışı başlasın)
Write-Host "4/4 Producer (Veri Kaynagi) Baslatiliyor..." -ForegroundColor Cyan
kubectl apply -f k8s/producer-job.yaml

Write-Host "------------------------------------------------" -ForegroundColor Green
Write-Host "KURULUM TAMAMLANDI!" -ForegroundColor Green
Write-Host "Dashboard'a gitmek icin tarayicinda su adresi ac:" -ForegroundColor White
Write-Host "http://localhost:30005" -ForegroundColor Yellow
Write-Host "------------------------------------------------" -ForegroundColor Green