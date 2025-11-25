import os
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)

GIGACHAT_KEY  = os.getenv('GIGACHAT_KEY')

PDF_DIR = "data"
CHROMA_DIR = "data/chroma_db"
COLLECTION_NAME = "rules"

model_path = "data/model_rnn_bidir.tar"
chroma_db_path = "data/chroma.sqlite3"