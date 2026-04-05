from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from core.prompts import contextualize_q_prompt, qa_prompt
from services.rag_service import rag_service
from services.llm_service import llm

chat_router = APIRouter()

# Chains
history_aware_retriever = create_history_aware_retriever(
    llm, rag_service.retriever, contextualize_q_prompt
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

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

@chat_router.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        print(f"Received question: {request.question}")
        print(f"Session ID: {request.session_id}")
        
        session_history = get_session_history(request.session_id).messages
        
        response = conversation.invoke(
            {"input": request.question, "chat_history": session_history},
            {"configurable": {"session_id": request.session_id}}
        )
        return JSONResponse(content={"response": response["answer"]})

    except Exception as e:
        print(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))
