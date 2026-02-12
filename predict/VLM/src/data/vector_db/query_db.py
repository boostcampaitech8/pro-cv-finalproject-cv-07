import requests


def search_news_by_embedding(host, port, collection_name, query_embedding, valid_dates, top_k=50):
    url = f"http://{host}:{port}/collections/{collection_name}/points/search"

    payload = {
        "vector": query_embedding,
        "limit": top_k,
        "with_payload": True, 
        "with_vector": True,
        "filter": {
            "must": [
                {
                    "key": "date",
                    "match": {"any": valid_dates}
                }
            ]
        }
    }

    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()