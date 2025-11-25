import os
import torch
import asyncio
from typing import TypedDict
from utilits.formatter_print import CustomPrint
from data.config import model_path, chroma_db_path

from langgraph.graph import StateGraph

from model import Model
from RNN import predict, train
from RAG import aquery_resp, process_pdfs_marker

class State(TypedDict):
    pass

class SupportRAG:
    def __init__(self):
        self.model: Model = self._load_model()

    def _load_model(self) -> Model:

        model_exists = os.path.exists(model_path)
        chroma_exists = os.path.exists(chroma_db_path) and os.listdir(chroma_db_path)

        if not model_exists:
            CustomPrint().warning("Модель RNN для классификации не найдена.")

        if not chroma_exists:
            CustomPrint().warning("Векторная база отсутствует.")

        if not model_exists:
            try:
                CustomPrint().info("Модель не найдена. Запустить обучение? Ctrl+C — отменить")
                input("Нажмите Enter для продолжения...")
            except KeyboardInterrupt:
                CustomPrint().info("Обучение отменено пользователем.")
                exit(0)

            m = Model()
            m.init_models()
            self.model = m

            ok = self.sync_run_training()
            if not ok:
                exit(1)

        m = Model()
        state = torch.load(model_path, map_location=m.device)
        m.rnn_model.load_state_dict(state)
        m.rnn_model.eval()

        return m
    
    def sync_run_training(self) -> bool:
        try:
            asyncio.run(self.train_model())
            CustomPrint().success("Обучение модели классификации завершено!")

            asyncio.run(process_pdfs_marker(self.model.embed_model))
            CustomPrint().success("Обработка документов для RAG завершена!")
            return True
        except Exception as e:
            CustomPrint().error(f"Ошибка при обучении: {e}")
            return False

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
        return await aquery_resp(
            embed_model=self.model.embed_model,
            question=question
        )

    async def create_workflow(self):
        pass

if __name__ == '__main__':
    rag = SupportRAG()
    question = "Как изменить пароль от личного кабинета?"
    answer = asyncio.run(rag.answer_rag(question))
    print(f"Вопрос: {question}\nОтвет: {answer}")