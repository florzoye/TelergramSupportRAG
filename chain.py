import os
import torch
from typing import TypedDict
from config import model_path, chroma_db_path

from langgraph.graph import StateGraph

from model import Model
from RNN import predict, train
from RAG import aquery_resp, read_all_pdfs

class State(TypedDict):
    pass

class SupportRAG:
    def __init__(self):
        self.model: Model = self._load_model()

    def _load_model(self) -> Model:
        if not os.path.exists(model_path):
            raise ValueError("⚠ Модель RNN не найдена. Сначала запустите обучение.")
         
        if not os.path.exists(chroma_db_path) or not os.listdir(chroma_db_path):
            print("⚠ Векторная база отсутствует. Создаю её из PDF...")
            ok = read_all_pdfs()
            if not ok:
                raise ValueError("Ошибка при создании векторной базы.")
            print("✔ Векторная база успешно создана.")
        
        m = Model()
        state = torch.load(model_path, map_location="cpu")
        m.rnn_model.load_state_dict(state)
        m.rnn_model.eval()

        return m

    async def train_model(self) -> bool:
        print("Запуск обучения RNN модели...")
        status = train(
            model=self.model.rnn_model,
            train_loader=self.model.data_loader,
            epochs=20
        )
        return status

    async def classify(self, phrase: str) -> bool:
        p = predict(
            phrase=phrase,
            model=self.model.rnn_model,
            transformer=self.model.transformer
        )
        return p.item() < 0.51

    async def answer_rag(self, question: str) -> str:
        return await aquery_resp(question)

    async def create_workflow(self):
        pass