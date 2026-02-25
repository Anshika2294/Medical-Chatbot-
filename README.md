# 🩺 SymptoSense – AI Powered Symptom Checker 

An AI-powered symptom checker chatbot built using LangChain, OpenAI, Pinecone, and Flask.
It uses Retrieval-Augmented Generation (RAG) to analyze user-entered symptoms and predict possible diseases using embedded medical knowledge.

⚠️ This system is for educational purposes only and does not replace professional medical advice.

## 🚀 Tech Stack

- Python
- LangChain
- Flask
- OpenAI GPT
- Pinecone (Vector Database)
- Docker
- AWS (EC2, ECR)
- GitHub Actions (CI/CD)

---

# 🧠 Local Setup Instructions

## STEP 1: Clone the Repository

```bash
git clone <https://github.com/Anshika2294/Medical-Chatbot-.git>
cd medical-chatbot
```

---

## STEP 2: Create Conda Environment

```bash
conda create -n medibot python=3.10 -y
conda activate medibot
```

---

## STEP 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## STEP 4: Create `.env` File

Create a `.env` file in the root directory and add:

```
PINECONE_API_KEY=pcsk_6zS5Q8_HrHbAovaaHNszkMazFeE7gPH5N3u3eosEAPhGXTWDobEiAKaGgG************
OPENAI_API_KEY=sk-proj-X0TUsl6o2IFfLb9SxLyXOK6feDqeK2mY2AxbxIsCqfqJlGTcNhbvG8q1ED2PnO7Pok7wAbCymUyfDpd9RmfaLD39T6zn2h1jqunGB2xR_*************
```

---

## STEP 5: Store Embeddings in Pinecone

```bash
python store_index.py
```

This will:
- Load PDFs
- Create embeddings
- Store them in Pinecone

---

## STEP 6: Run the Application

```bash
python app.py
```

Now open your browser:

```
http://localhost:5000
```

---

# 🐳 Docker Deployment

## 1️⃣ Build Docker Image

```bash
docker build -t medicalbot .
```

---

## 2️⃣ Tag Docker Image

```bash
docker tag medicalbot:latest <your-ecr-uri>
```

Example:
```
315865595366.dkr.ecr.us-east-1.amazonaws.com/medicalbot
```

---

## 3️⃣ Push Image to AWS ECR

Authenticate Docker with AWS:

```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 315865595366.dkr.ecr.us-east-1.amazonaws.com
```

Push Image:

```bash
docker push 315865595366.dkr.ecr.us-east-1.amazonaws.com/medicalbot
```

---

# ☁️ AWS Deployment Steps

## 1️⃣ Create IAM User

Attach Policies:

- AmazonEC2FullAccess
- AmazonEC2ContainerRegistryFullAccess

---

## 2️⃣ Create ECR Repository

Save the repository URI.

---

## 3️⃣ Launch EC2 Instance (Ubuntu)

---

## 4️⃣ Install Docker on EC2

```bash
sudo apt-get update -y
sudo apt-get upgrade -y

curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

sudo usermod -aG docker ubuntu
newgrp docker
```

---

## 5️⃣ Pull Docker Image in EC2

```bash
docker pull <your-ecr-uri>
```

---

## 6️⃣ Run Docker Container

```bash
docker run -d -p 5000:5000 \
-e PINECONE_API_KEY=your_key \
-e OPENAI_API_KEY=your_key \
medicalbot
```

---

# 🔁 CI/CD with GitHub Actions

## Configure Self-Hosted Runner on EC2

Go to:

```
GitHub → Settings → Actions → Runners → New Self-Hosted Runner
```

Follow the provided commands in EC2 terminal.

---

## Add GitHub Secrets

Go to:

```
Repository → Settings → Secrets and Variables → Actions
```

Add:

- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACCESS_KEY
- AWS_DEFAULT_REGION
- ECR_REPO
- PINECONE_API_KEY
- OPENAI_API_KEY

---

# 🏗 Deployment Workflow Overview

1. Build Docker image
2. Push to AWS ECR
3. EC2 pulls latest image
4. Container runs automatically

---

# 📌 Features

- PDF-based medical knowledge
- RAG pipeline
- Vector search with Pinecone
- Cloud deployment ready
- CI/CD automation

---

# 👩‍💻 Author

Anshika Shukla  
B.Tech Information Technology  
AI & ML Enthusiast 🚀
