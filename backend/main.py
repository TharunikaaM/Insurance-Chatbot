from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.chat import chat_router

app = FastAPI(title="Insurance Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)

@app.get("/")
async def home():
    return {"message": "Insurance Chatbot Backend is running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
