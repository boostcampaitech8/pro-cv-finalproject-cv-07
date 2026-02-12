from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from qdrant_client.models import VectorParams, Distance


def connect_vector_db(host, port):
    client = QdrantClient(host=host, port=port)
    return client


def create_collection(db_client, collection_name, dimension=512):
    db_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
    )
    

def push_data(db_client, collection_name, df, batch_size=1000):
    points_batch = []
    
    for idx, row in df.iterrows():
        points_batch.append(
            PointStruct(
                id=row['id'],
                vector=row['article_embedding'],
                payload={'date': row['trade_date'].strftime('%Y-%m-%d'),
                         'title': row['title'],
                         'description': row['description']}
                )
            )

        if len(points_batch) >= batch_size:
            db_client.upsert(collection_name=collection_name, points=points_batch)
            points_batch = []

    if points_batch:
        db_client.upsert(collection_name=collection_name, points=points_batch)

    print("모든 포인트 업로드 완료")