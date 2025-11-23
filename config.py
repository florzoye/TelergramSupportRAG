import os
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)

OPENAI_KEY = os.getenv("OPENAI_KEY", "")


model_path = "data/model_rnn_bidir.tar"
chroma_db_path = "data/chroma.sqlite3"