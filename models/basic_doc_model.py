from pydantic import BaseModel

from typing import List

class DocModel(BaseModel):
    sentences: List[str]