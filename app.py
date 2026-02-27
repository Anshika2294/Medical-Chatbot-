from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import system_prompt
import os

# Initialize Flask
app = Flask(__name__)

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Load Embeddings
embeddings = download_hugging_face_embeddings()

# Connect to Existing Pinecone Index
index_name = "symptosense"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Create Retriever
retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# Load Chat Model
chat_model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3,
    api_key=GROQ_API_KEY
)


# Create Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Create RAG Chain
question_answer_chain = create_stuff_documents_chain(
    chat_model,
    prompt
)

rag_chain = create_retrieval_chain(
    retriever,
    question_answer_chain
)

# Home Route
@app.route("/")
def index():
    return render_template("chat.html")

# Chat Route
@app.route("/get", methods=["POST"])
def chat():
    try:
        user_message = request.form["msg"]
        print("User:", user_message)

        response = rag_chain.invoke({
            "input": user_message
        })

        print("Raw Response:", response)

        return response["answer"]

    except Exception as e:
        print("ERROR:", str(e))
        return "Something went wrong. Please try again."

# Run App
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080 )