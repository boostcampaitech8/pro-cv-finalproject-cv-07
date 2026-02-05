from google.cloud import storage
import os
from dotenv import load_dotenv

load_dotenv()

def upload_to_gcs(file_object, destination_blob_name):
    storage_client = storage.Client.from_service_account_json(os.getenv('GOOGLE_APPLICATION_CREDENTIALS'))
    bucket = storage_client.bucket(os.getenv('BUCKET_NAME'))
    blob = bucket.blob(destination_blob_name)
    
    if hasattr(file_object, 'seek'):
        file_object.seek(0)
        
    blob.upload_from_file(file_object, content_type="image/jpeg")
    
    url = f"https://storage.googleapis.com/{os.getenv('BUCKET_NAME')}/{destination_blob_name}"
    
    if isinstance(url, bytes):
        url = url.decode('utf-8')
    
    url = str(url).replace("b'", "").replace("'", "")
    
    return url


def delete_from_gcs(file_url: str):
    storage_client = storage.Client.from_service_account_json(os.getenv('GOOGLE_APPLICATION_CREDENTIALS'))
    bucket = storage_client.bucket(os.getenv('BUCKET_NAME'))

    if isinstance(file_url, bytes):
        file_url = file_url.decode("utf-8")

    try:
        prefix = f"https://storage.googleapis.com/{os.getenv('BUCKET_NAME')}/"
        if file_url.startswith(prefix):
            path = file_url[len(prefix):]
        else:
            path = file_url

        blob = bucket.blob(path)
        blob.delete()
        print(f"GCS 파일 삭제 완료: {path}")
    except Exception as e:
        print(f"GCS 파일 삭제 실패: {e}")