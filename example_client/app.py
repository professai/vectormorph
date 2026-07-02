import os

import requests
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel

VECTORMORPH_URL = os.environ.get("VECTORMORPH_URL", "http://vectormorph-example:4440")

openai_client = OpenAI()  # Reads OPENAI_API_KEY from the environment


def get_embedding(text: str) -> list:
    response = openai_client.embeddings.create(
        input=text,
        model="text-embedding-3-small",
    )
    return response.data[0].embedding


app = FastAPI(
    title="VectorMorph Client",
    description="A client for VectorMorph.",
)


def vectormorph_headers() -> dict:
    return {"Authorization": f"Bearer {os.environ.get('BEARER_TOKEN')}"}


class AddRequest(BaseModel):
    summary_text: str = "This is a summary of the document."
    document_text: str = "This is the full text of the document."


class SearchRequest(BaseModel):
    query_text: str = "What is this document about?"
    k: int = 10


@app.post("/data/add/")
async def add_data(request: AddRequest):
    summary_embedding = get_embedding(request.summary_text)
    document_embedding = get_embedding(request.document_text)

    response = requests.post(
        f"{VECTORMORPH_URL}/add/",
        json={
            "summary_vector": summary_embedding,
            "document_vector": document_embedding,
        },
        headers=vectormorph_headers(),
    )
    if not response.ok:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    return response.json()


@app.post("/data/search/")
async def search_data(request: SearchRequest):
    query_embedding = get_embedding(request.query_text)

    response = requests.post(
        f"{VECTORMORPH_URL}/search/",
        json={"query_vector": query_embedding, "k": request.k},
        headers=vectormorph_headers(),
    )
    if not response.ok:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    return response.json()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=80)
