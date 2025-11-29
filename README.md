# 🚀 Büyük Veri Analitik Sistemi (Big Data Analytics Pipeline)

Bu proje, **Kubernetes** üzerinde çalışan, mikroservis mimarisine sahip uçtan uca bir Büyük Veri Analitik Sistemidir.

Gerçek zamanlı veri akışı, işleme, makine öğrenmesi tahmini ve canlı görselleştirme aşamalarını içerir. Proje, "Store Sales Time Series Forecasting" veri setini kullanarak mağaza satışlarını simüle eder ve anlık tahminler üretir.

---

## 🏗️ Sistem Mimarisi

Sistem, Docker konteynerleri içinde çalışan 6 temel bileşenden oluşur ve Kubernetes (Orchestrator) tarafından yönetilir.

| Bileşen                 | Teknoloji                             | Görevi                                                                    |
| :---------------------- | :------------------------------------ | :------------------------------------------------------------------------ |
| **Data Source**         | `Python Producer`                     | `train.csv` verisini okur ve Kafka'ya canlı akış (stream) olarak basar.   |
| **Message Broker**      | `Apache Kafka` & `Zookeeper`          | Veri dağıtımını ve kuyruklama işlemini yönetir.                           |
| **Storage (Raw)**       | `Raw Consumer`                        | Kafka'dan gelen ham veriyi işlenmeden `MongoDB`'ye yedekler.              |
| **Stream Processor**    | `Apache Spark (Structured Streaming)` | Veriyi canlı okur, ML modelinden geçirir ve işlenmiş sonucu yazar.        |
| **Storage (Processed)** | `MongoDB`                             | Hem ham verilerin hem de tahmin sonuçlarının saklandığı NoSQL veritabanı. |
| **Monitoring**          | `Streamlit Dashboard`                 | Veritabanından sonuçları canlı çeker ve grafiksel olarak sunar.           |

---

## 🛠️ Gereksinimler

Projeyi çalıştırmadan önce bilgisayarınızda şunların yüklü olması gerekir:

1. **Docker Desktop:** (Ayarlardan Kubernetes aktif edilmiş olmalı).
2. **PowerShell:** (Windows için yönetim scriptlerini çalıştırmak amacıyla).
3. **RAM:** Docker Desktop için en az 4GB (Tercihen 6GB) RAM ayrılmış olmalıdır.

---

## 🚀 Kurulum ve Çalıştırma (Adım Adım)

Proje, tek bir script ile otomatik olarak kurulup başlatılabilir.

### 1. Hazırlık

Proje klasörüne terminal (PowerShell - Yönetici Modunda) üzerinden gidin:

```powershell
cd C:\Users\KullaniciAdi\ProjeKlasoru
```

### 2. Başlatma (Deploy)

Kurulum scriptini çalıştırın. Bu işlem; eski kurulumları temizler, Docker imajlarını derler ve Kubernetes podlarını başlatır.

```powershell
.\deploy.ps1
```

_(Not: İlk çalıştırmada imajların inmesi internet hızına bağlı olarak 3-5 dakika sürebilir.)_

### 3. Dashboard'a Erişim (Sonuçları Görme)

Terminalde "KURULUM TAMAMLANDI" yazısını gördükten sonra tarayıcınızdan şu adrese gidin:

👉 **http://localhost:30005**

_(Eğer sayfa açılmazsa, port yönlendirmesi için aşağıdaki komutu kullanın ve http://localhost:8501 adresine gidin:)_

```powershell
kubectl port-forward service/dashboard-service 8501:8501
```

---

## 💻 Terminal Komutları (Sunum İçin)

Sunum sırasında sistemi yönetmek ve kanıtlamak için kullanabileceğiniz kritik komutlar:

### A. Sistemin Çalıştığını Kontrol Etme

```powershell
kubectl get pods
```

### B. Canlı Logları İzleme (Kanıt Gösterme)

**1. Veri Üreticisi (Producer):**

```powershell
kubectl logs -f job/producer-job
```

_(Çıktı: "Sent 100 records..." şeklinde akmalı)_

**2. Spark İşleyici (Processor):**

```powershell
kubectl logs -f job/spark-job
```

_(Çıktı: Model tahmin loglarını içermeli)_

**3. Ham Veri Kaydedici (Raw Consumer):**

```powershell
kubectl logs -f deployment/raw-consumer
```

_(Çıktı: "Ham veri kaydedildi..." yazmalı)_

### C. Veri Akışını Yeniden Başlatma

```powershell
kubectl delete job producer-job
kubectl apply -f k8s/producer-job.yaml
```

---

## 📂 Proje Dosya Yapısı

```text
📦 Homework
 ┣ 📂 dataset           # Veri seti (train.csv)
 ┣ 📂 producer          # Veri kaynağı simülasyon kodları
 ┣ 📂 raw_consumer      # Ham veriyi kaydeden Python scripti
 ┣ 📂 spark             # Spark Streaming ve ML kodları
 ┣ 📂 dashboard         # Streamlit görselleştirme arayüzü
 ┣ 📂 k8s               # Kubernetes konfigürasyon (YAML) dosyaları
 ┣ 📜 deploy.ps1        # Otomatik kurulum scripti
 ┗ 📜 README.md         # Proje dokümantasyonu
```

---

## ⚠️ Sık Karşılaşılan Sorunlar ve Çözümleri

**Soru:** Dashboard açılmıyor, "Connection Refused" hatası alıyorum.  
**Çözüm:** `kubectl port-forward service/dashboard-service 8501:8501` komutunu çalıştırın ve tarayıcıdan `localhost:8501` adresini deneyin.

---

**Soru:** Podlar "Pending" durumunda kalıyor.  
**Çözüm:** Docker Desktop → Resources → Memory kısmını 4GB veya üzerine çıkarın.

---

**Soru:** Bilgisayarı kapatıp açtım, projeyi nasıl tekrar başlatırım?  
**Çözüm:** Sadece `.\deploy.ps1` komutunu tekrar çalıştırmanız yeterlidir.
