# 🩺 SymptoSense – AI Powered Symptom Checker 

An AI-powered symptom checker chatbot built using LangChain, OpenAI, Pinecone, and Flask.
It uses Retrieval-Augmented Generation (RAG) to analyze user-entered symptoms and predict possible diseases using embedded medical knowledge.

⚠️ This system is for educational purposes only and does not replace professional medical advice.

## 🚀 Tech Stack

- Python
- LangChain
- Flask
- GROQ
- Pinecone (Vector Database)
 GitHub Actions (CI/CD)

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
PINECONE_API_KEY=***************************************************************************
GROQ_API_KEY=*******************************************************************************

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
http://127.0.0.1:8080/?
```

---

# 📌 Features

- PDF-based Symptom checker
- RAG pipeline
- Vector search with Pinecone
- Cloud deployment ready
- CI/CD automation

---

# 👩‍💻 Author

Anshika Shukla  
B.Tech Information Technology  
AI & ML Enthusiast 🚀
