import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings

from RNN import WordsRNN, PhraseDataset, collate_fn

class Model:
    def __init__(self):
        self.device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dataset_path_true: str = "data/train_data_true"
        self.dataset_path_false: str = "data/train_data_false"

    def init_models(self):
        self.rnn_model = WordsRNN(384, 1)
        self.transformer = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.phrase_dataset = PhraseDataset(
            self.dataset_path_true,
            self.dataset_path_false,
            self.transformer
        )
        self.data_loader = DataLoader(
            self.phrase_dataset, 
            shuffle=True, 
            batch_size=8, 
            collate_fn=collate_fn
        )
        self.embed_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': self.device},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.rnn_model.to(self.device)