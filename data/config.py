import os
import base64
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)

GIGACHAT_AUTH_KEY = os.getenv('GIGACHAT_AUTH_KEY')
GIGACHAT_SCOPE = "GIGACHAT_API_PERS" 
BOT_TOKEN = os.getenv('BOT_TOKEN')

PDF_DIR = "data"
CHROMA_DIR = "data/chroma_db"
COLLECTION_NAME = "rules"

model_path = "data/model_rnn_bidir.tar"
chroma_db_path = "data/chroma_db"