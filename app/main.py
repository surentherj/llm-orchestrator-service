from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI

from app.api.chat_api import router as chat_router

app = FastAPI(
    title="LLM Orchestrator Service",
    description="Multi-model RAG Orchestrator",
    version="1.0.0"
)

# Register routes
app.include_router(chat_router)


@app.get("/")
async def root():

    return {
        "status": "running"
    }