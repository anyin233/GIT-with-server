import uvicorn
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from models.basic_doc_model import DocModel
import json
from run_inf_task import run_inf_task

app = FastAPI()

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/ee")
async def event_extract(doc: DocModel):
    token_id = "00000000"
    sentences = doc.sentences
    with open("./Data/dev/test.json", 'w') as f:
        json.dump([[token_id, {"sentences": sentences}]], f, ensure_ascii=False)
    return run_inf_task()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)