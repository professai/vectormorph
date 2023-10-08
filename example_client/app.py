import openai
import os
from fastapi import FastAPI, WebSocket, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from starlette.requests import Request
import requests

# Assume you have a function to get embeddings from OpenAI
def get_embedding(text):
    # This is a simplified example; the actual code to get an embedding from OpenAI may be different
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    embeddings = response['data'][0]['embedding']
    return embeddings

app = FastAPI(
    title="VectorMorph Client",
    description="A client for VectorMorph.",
)

class RequestBody(BaseModel):
    summary_text: str = "This is a summary of the document."
    document_text: str = "This is the full text of the document."

@app.post("/data/add/")
async def add_data(request: RequestBody):
    summary_embedding = get_embedding(request.summary_text)
    document_embedding = get_embedding(request.document_text)

    # URL of your VectorMorph server's /add/ endpoint
    url = 'http://vectormorph-example:4440/add/'

    # Your bearer token for authentication
    headers = {
        'Authorization': f'Bearer {os.environ.get("BEARER_TOKEN")}'
    }

    # Data to send to the VectorMorph server
    data = {
        'summary_vector': [summary_embedding],
        'document_vector': [document_embedding]
    }

    # Send a POST request to the VectorMorph server
    response = requests.post(url, json=data, headers=headers)
    return response.json()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)
