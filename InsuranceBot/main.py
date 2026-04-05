from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel
import os
from openai import OpenAI

app = FastAPI()

# Add CORS middleware to allow requests from React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can specify your frontend URL here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from pathlib import Path
embeddings = GPT4AllEmbeddings()

embedding_file_path = "savedEmbeddings"
docs = []

if os.path.exists(embedding_file_path):
    vectorstore = Chroma(
        embedding_function = embeddings,
        persist_directory = embedding_file_path
    )
else:
    policy_directory = Path(r"C:\Users\kamal\OneDrive\Desktop\InsuranceBot\Policies")
    files = list(policy_directory.glob("*.pdf"))
    for file in files:
        loader = PyPDFLoader(str(file))
        docs.extend(loader.load())
    
    vectorstore = Chroma.from_documents(
        documents = docs,
        embedding = embeddings,
        persist_directory = embedding_file_path
    )
print("Embeddings Completed")

retriever = vectorstore.as_retriever(search_type="similarity", kwargs={"k":5})

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = "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"
)


llm = client.chat.completions.create(
  model="nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
  messages=[{"role":"user","content":""}],
  temperature=1.00,
  top_p=0.01,
  max_tokens=1024,
  stream=True
)

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

qa_system_prompt = """
You are an highly professional and knowledgeable Insurance Bot. Your task is to answer user questions with precision, 
using only the exact provided context from the PDF. 

Follow these guidelines to answer as a professional insurance agent:
- Use **only** the content found in the documents {{docs}}. If the exact answer is not available, respond with: 
  "The document does not provide this information." 
- If the user's question is unclear or outside the scope of the document, 
  politely let them know and suggest they ask another question within the document's context.
- Always be polite, concise, and ensure the answer is accurate.
- If the information in the PDF is limited, provide the closest matching answer and inform the user about any limitations.
- Greet the user professionally if they greet you, before answering.
- If you are gonna provide table output, add grid to it.
- please provide the answer properly

Context:
{context}

User's Question:
{input}
"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Store chat history in a dictionary
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversation = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)

class QuestionRequest(BaseModel):
    question: str
    session_id: str

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        # Log the received request
        print(f"Received question: {request.question}")
        print(f"Session ID: {request.session_id}")
        
        # Get session history (log the session history)
        session_history = get_session_history(request.session_id).messages
        print(f"Session History: {session_history}")
        
        # Execute the chain and log the response
        response = conversation.invoke(
            {"input": request.question, "chat_history": session_history},
            {"configurable": {"session_id": request.session_id}}
        )
        print(f"Chain Response: {response}")

        # Return the chain response in the JSON
        return JSONResponse(content={"response": response["answer"]})

    except Exception as e:
        # Log the exception and raise an HTTP error
        print(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def home():
    return {"message": "Server is running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
