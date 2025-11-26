import os
import torch
import asyncio
from typing import TypedDict, Literal
from utilits.formatter_print import CustomPrint
from data.config import model_path, chroma_db_path

from langgraph.graph import StateGraph
from langgraph.constants import START, END

from .model import Model
from .RNN import predict, train
from .RAG import aquery_resp, process_pdfs_marker

class State(TypedDict):
    question: str
    is_relevant: bool | None
    model_response: str

class SupportRAG:
    def __init__(self):
        self.model: Model = self._load_model()
        self.workflow = self._build_workflow_sync()

    def _load_model(self) -> Model:
        model_exists = os.path.exists(model_path)
        chroma_exists = os.path.exists(chroma_db_path) and os.listdir(chroma_db_path)

        rnn = not model_exists
        rag = not chroma_exists

        self.model = Model().init_models()

        if rnn or rag:
            try:
                self.choose_train(rnn, rag)
            except KeyboardInterrupt:
                CustomPrint().info("Обучение отменено пользователем.")
                exit(0)

        if model_exists:
            state = torch.load(model_path, map_location=self.model.device)
            self.model.rnn_model.load_state_dict(state)
            self.model.rnn_model.eval()
            CustomPrint().success("RNN модель успешно загружена.")

        if chroma_exists:
            CustomPrint().success("Векторная база RAG загружена.")
        elif rag:
            pass

        return self.model

    
    def _sync_run_training_rnn(self) -> bool:
        try:
            asyncio.run(self.train_model())
            CustomPrint().success("Обучение модели классификации завершено!")
        except Exception as e:
            CustomPrint().error(f"Ошибка при обучении: {e}")
            return False
        
    def _sync_run_training_rag(self) -> bool:
        try:
            process_pdfs_marker(self.model.embed_model)
            CustomPrint().success("Обработка документов для RAG завершена!")
            return True
        except Exception as e:
            CustomPrint().error(f"Ошибка при обучении: {e}")
            return False

    async def _classify(self, phrase: str) -> bool:
        p = predict(
            phrase=phrase,
            model=self.model.rnn_model,
            transformer=self.model.transformer
        )
        print(p.item())
        return p.item() < 0.51
    
    async def _answer_rag(self, question: str) -> str:
        return await aquery_resp(
            embed_model=self.model.embed_model,
            question=question
        )

    def choose_train(self, rnn: bool, rag: bool):
        try:
            if rnn:
                CustomPrint().info("Модель классификации не найдена. Запустить обучение? Ctrl+C — отменить")
                input("Нажмите Enter для продолжения...")
                self._sync_run_training_rnn()
            if rag:
                CustomPrint().info("RAG модель не найдена. Запустить обучение? Ctrl+C — отменить")
                input("Нажмите Enter для продолжения...")
                self._sync_run_training_rag()
        except Exception as e:
            CustomPrint().error(f'Ошикба во время тренировки моделей - {e}')
            
    async def train_model(self) -> bool:
            print("Запуск обучения RNN модели...")
            status = train(
                model=self.model.rnn_model,
                train_loader=self.model.data_loader,
                epochs=20
            )
            return status
    
    async def classify(self, state: State) -> State:
        is_relevant = await self._classify(state["question"])
        return {
            "question": state["question"],
            "is_relevant": is_relevant,
            "model_response": ""
        }

    async def search_relevant_data(self, state: State) -> State:
        answer = await self._answer_rag(state["question"])
        return {
            "question": state["question"],
            "is_relevant": state["is_relevant"],
            "model_response": answer
        }

    async def reject_message(self, state: State) -> State:
        return {
            "question": state["question"],
            "is_relevant": state["is_relevant"],
            "model_response": "Ваш вопрос не относится к поддерживаемой теме. Пожалуйста, задайте вопрос по теме проекта."
        }

    def router(self, state: State) -> Literal["search_relevant_data", "reject_message"]:
        if state["is_relevant"] is True:
            return "search_relevant_data"
        return "reject_message"

    def _build_workflow_sync(self):
        workflow = StateGraph(State)

        workflow.add_node("classify_prompt", self.classify)
        workflow.add_node("search_relevant_data", self.search_relevant_data)
        workflow.add_node("reject_message", self.reject_message)

        workflow.add_conditional_edges(
            "classify_prompt",
            self.router,
            {
                "search_relevant_data": "search_relevant_data",
                "reject_message": "reject_message"
            }
        )

        workflow.add_edge(START, "classify_prompt")
        workflow.add_edge("search_relevant_data", END)
        workflow.add_edge("reject_message", END)

        return workflow.compile()
    
    async def process_question(self, question: str) -> str:
        result = await self.workflow.ainvoke({
            "question": question,
            "is_relevant": None,
            "model_response": ""
        })
        return result["model_response"]
    

if __name__ == '__main__':
    rag = SupportRAG()

    async def main():
        while(True):
            try:
                question = str(input('Введите ваш вопрос: '))
                answer = await rag.process_question(question)
                print(f"Вопрос: {question}")
                print(f"Ответ: {answer}")
            except KeyboardInterrupt:
                print('Пока')
                break

    asyncio.run(main())