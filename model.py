from pydantic import BaseModel
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer

from RNN import WordsRNN, PhraseDataset, collate_fn

class Model(BaseModel):
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    dataset_path_true: str = "data/train_data_true"
    dataset_path_false: str = "data/train_data_false"

    rnn_model: WordsRNN = WordsRNN(384, 1)
    transformer: SentenceTransformer = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    phrase_dataset: PhraseDataset = PhraseDataset(
        dataset_path_true,
        dataset_path_false,
        transformer=transformer 
    )
    data_loader: DataLoader = DataLoader(
        dataset=phrase_dataset,
        shuffle=True,
        batch_size=8,
        collate_fn=collate_fn
    )

    