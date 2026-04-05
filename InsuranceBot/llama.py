from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_ollama import ChatOllama
#from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel

app = FastAPI()

# Add CORS middleware to allow requests from React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can specify your frontend URL here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

files=[
       r"Policies\Accidental Death Benefit Rider Plan.pdf",
       r"Policies\Cash Back Plan.pdf",
       r"Policies\Csc Shubhlabh Plan.pdf",
       r"Policies\Guaranteed Protection Plus Plan.pdf",
       r"Policies\Elite Term Plan.pdf",
       r"Policies\Gold Plan.pdf",
       r"Policies\Guaranteed Benefit Plan.pdf",
       r"Policies\Insurance Khata Plan.pdf",
       r"Policies\Little Champ Plan.pdf",
       r"Policies\Long Guaranteed Income Plan.pdf",
       r"Policies\Maha Jeevan Plan.pdf",
       r"Policies\Micro Bachat Plan.pdf",
       r"Policies\Money Balance Plan.pdf",
       r"Policies\Life Plan.pdf",
       r"Policies\Pos Cash Back Plan.pdf",
       r"Policies\Radiance Smart Investment Plan.pdf",
       r"Policies\Saral Bachat Bima Plan.pdf",
       r"Policies\Saral Jeevan Bima Plan.pdf",
       r"Policies\Simple Benefit Plan.pdf",
       r"Policies\Single Premium Plan.pdf",
       r"Policies\Smart Pay Plan.pdf",
       r"Policies\Smart Save Plan.pdf",
       r"Policies\Tulip Plan.pdf",
       r"Policies\Wealth Maximizer Plan.pdf"
       ]

docs=[]
for file_path in files:
    loader_pdf = PyPDFLoader(file_path)
    docs.extend(loader_pdf.load())

vectorstore = Chroma.from_documents(docs, embedding=GPT4AllEmbeddings())
retriever = RunnableLambda(vectorstore.similarity_search).bind(k=5)

qa_system_prompt = """
You are an highly professional and knowledgeable Insurance Bot. Your task is to answer user questions with precision, 
using only the exact provided context from the PDF.

Available policies: Accidental Death Benefit Rider Plan, Cash Back Plan, Csc Shubhlabh Plan, Guaranteed Protection Plus Plan, Elite Term Plan, 
Gold Plan, Guaranteed Benefit Plan, Insurance Khata Plan, Little Champ Plan, Long Guaranteed Income Plan, Maha Jeevan Plan, Micro Bachat Plan, 
Money Balance Plan, Life Plan, Pos Cash Back Plan, Radiance Smart Investment Plan, Saral Bachat Bima Plan, Saral Jeevan Bima Plan, 
Simple Benefit Plan, Single Premium Plan, Smart Pay Plan, Smart Save Plan, Tulip Plan, Wealth Maximizer Plan

Question: I am 40 and wants moderate coverage with monthly premiums, would pay Rs 4,000-5,000 every month. He wants a life insurance protection and flexible investments, ensuring his family’s financial security while growing his wealth. What plan do you suggest?
Answer: IndiaFirst Life Radiance Smart Investment Plan

Question: I am 10 years old and can pay Rs. 1600 every month from my pocket money. What plan do you suggest me?
Answer: IndiaFirst Life Smart Pay Plan

Question: I am 35 years old. I can save Rs. 2500 every year from my income and i am searching for a plan with long term financial security for my family. What plan do  you suggest for me?
Answer: IndiaFirst Life Guaranteed Protection Plus Plan 

Question: I am 45 years old. I can pay Rs. 530 every month and i want a periodic returns for my short term financial goals. What plan do you suggest me?
Answer: IndiaFirst Life POS Cash Back Plan

Question: As a 35 year old single parent, Priya wanted to secure her daughter’s future, especially for education.  She can pay Rs. 1400 monthly. which plan do u suggest her  giving her peace of mind that her child’s dreams would be protected, no matter what?
Answer: IndiaFirst Life Cash Back Plan

Follow these guidelines to answer as a professional insurance agent:
- Use **only** the content found in the documents{{Policy Details:
1. **Life Accidental Death Benefit Rider**  
   - Benefits: Lump sum payout on accidental death (100% ADB sum assured)
   - Eligibility: Age 18-70 years
   - Premium Payment Term: Single, Regular, or Limited Pay (2-12 years)
   - Claim Process: Submit claim form, policy documents, proof of accidental death

2. **Life Cash Back Plan**  
   - Benefits: Periodical payouts, maturity payout (60% Sum assured), death benefit
   - Eligibility: Age 15-55 years, max maturity age 70
   - Premium Payment Term: Limited pay (5, 7, or 10 years)
   - Claim Process: Submit claim form, policy documents, proof of death

3. **CSC Shubhlabh Plan**  
   - Benefits: Lump sum maturity benefit, life cover on death
   - Eligibility: Age 18-55 years, max cover age 65
   - Premium Payment Term: 10 or 15 years
   - Claim Process: Submit claim form, policy documents, proof of death

4. **Life Guaranteed Protection Plus Plan**  
   - Benefits: Lump sum on death or terminal illness, optional returns
   - Eligibility: Age 18-65 years (varies by option)
   - Premium Payment Term: Single, Limited, or Regular pay (5-47 years)
   - Claim Process: Submit claim form, policy documents, identity proof

5. **Life Elite Term Plan**  
   - Benefits: Life cover up to age 99, lump sum on death
   - Eligibility: Age 18-65 years
   - Premium Payment Term: Regular pay
   - Claim Process: Submit claim form, policy documents, identity proof

6. **Life Guarantee of Life Dreams Plan**  
   - Benefits: Guaranteed income, life cover, loyalty benefits, tax benefits
   - Eligibility: Entry age 90 days, max age 50-60 years, maturity up to 90 years
   - Premium Payment Term: 6, 8, or 10 years
   - Claim Process: Submit claim form, policy documents, identity proof

7. **Life Guaranteed Benefit Plan**  
   - Benefits: Income or Lump sum Benefit, life cover, waiver of premium
   - Eligibility: Age 8-60 years, max maturity age 76 years
   - Premium Payment Term: 5, 6, or 7 years
   - Claim Process: Submit claim form, policy documents, identity proof

8. **Life "Insurance Khata" Plan**  
   - Benefits: Lump sum on death, assured maturity benefit
   - Eligibility: Age 18-45 years
   - Premium Payment Term: Single pay
   - Claim Process: Submit claim form, policy documents, identity proof

9. **Life Little Champ Plan**  
   - Benefits: Lump sum on death, guaranteed term payouts
   - Eligibility: Age 21-45 years
   - Premium Payment Term: Limited pay (7-14 years)
   - Claim Process: Submit claim form, policy documents, identity proof

10. **Life Long Guaranteed Income Plan**  
    - Benefits: Lump sum on death, guaranteed income benefits
    - Eligibility: Age 8-60 years
    - Premium Payment Term: Limited pay (5/6/7 years)
    - Claim Process: Submit claim form, policy documents, identity proof}}. If the exact answer is not available, respond with: 
  "The document does not provide this information." 
- If the user's question is unclear or outside the scope of the document, 
  politely let them know and suggest they ask another question within the document's context.
- Always be polite, concise, and ensure the answer is accurate.
- If the information in the PDF is limited, provide the closest matching answer and inform the user about any limitations.
- Greet the user professionally if they greet you, before answering.

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

llm = ChatOllama(model="llama3.2")

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

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Store chat history in a dictionary
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
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
        print(f"Incoming request: {request.model_dump()}")  
        
        session_history = get_session_history(request.session_id).messages
        response = conversational_rag_chain.invoke(
            {"input": request.question, "chat_history": session_history},
            {"configurable": {"session_id": request.session_id}}  
        )
        return JSONResponse(content={"response": response['answer']})
    except ValueError as e:
        print(f"Error invoking chain: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def home():
    return {"message": "Server is running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)