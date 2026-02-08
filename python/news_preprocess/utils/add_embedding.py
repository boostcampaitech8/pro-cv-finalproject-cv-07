import os
import ast
import numpy as np
import pandas as pd
import json
import boto3
from tqdm import tqdm
from copy import deepcopy

class TitanEmbeddings(object):
    accept = "application/json"
    content_type = "application/json"

    def __init__(
        self,
        model_id="amazon.titan-embed-text-v2:0",
        access_key_id=None,
        secret_access_key=None,
        session_token=None,
        region_name="us-east-1",
    ):
        client_kwargs = {
            "service_name": "bedrock-runtime",
            "region_name": region_name,
        }
        if access_key_id and secret_access_key:
            client_kwargs["aws_access_key_id"] = access_key_id
            client_kwargs["aws_secret_access_key"] = secret_access_key
        if session_token:
            client_kwargs["aws_session_token"] = session_token

        self.bedrock = boto3.client(**client_kwargs)
        self.model_id = model_id
    def __call__(self, text, dimensions, normalize=True):
        body = json.dumps({
            "inputText": text,
            "dimensions": dimensions,
            "normalize": normalize
        })
        response = self.bedrock.invoke_model(
            body=body, modelId=self.model_id, accept=self.accept, contentType=self.content_type
        )
        response_body = json.loads(response.get('body').read())
        return response_body['embedding']


def add_article_embedding(
    texts: list[str],
    model_id: str = "amazon.titan-embed-text-v2:0",
    dimensions: int = 512,
    normalize: bool = True,
    access_key_id = None, 
    secret_access_key = None,
    session_token = None,
):
    titan = TitanEmbeddings(
        model_id=model_id,
        access_key_id=access_key_id,
        secret_access_key=secret_access_key,
        session_token=session_token,
    )
    embs = []
    for t in tqdm(texts, desc="Embedding generation"):
        if t is None:
            t = ""
        embs.append(titan(str(t), dimensions=dimensions, normalize=normalize))
    return embs