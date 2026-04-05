from openai import OpenAI
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableLambda
from langchain_core.language_models.chat_models import BaseChatModel
from core.config import settings

class DummyLLM:
    """A wrapper for the OpenAI client if we want it to act like a LangChain ChatModel temporarily,
    but we are better off using standard LangChain wrappers if available. 
    For NVIDIA, we'll wrap the OpenAI client specifically for the chains.
    """
    pass

def get_llm():
    if settings.LLM_PROVIDER == "OLLAMA":
        print("Using OLLAMA as LLM provider.")
        return ChatOllama(model="llama3.2")
    
    elif settings.LLM_PROVIDER == "NVIDIA":
        print("Using NVIDIA endpoint as LLM provider.")
        # Currently, LangChain Nvidia endpoints exist but to mimic the old main.py exactly:
        from langchain_openai import ChatOpenAI
        
        return ChatOpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=settings.NVIDIA_API_KEY,
            model="nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
            temperature=1.00,
            model_kwargs={"top_p": 0.01},
            max_tokens=1024
        )
    else:
        raise ValueError(f"Unknown LLM Provider: {settings.LLM_PROVIDER}")

llm = get_llm()
