import os
import glob
import torch
import asyncio
from config import GIGACHAT_KEY, PDF_DIR, CHROMA_DIR, COLLECTION_NAME

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_gigachat import GigaChat
from langchain_core.documents import Document

 
def process_pdfs_marker():
    """
    Обрабатывает PDF c помощью Marker OCR и индексирует в Chroma.
    """
    if not os.path.exists(PDF_DIR):
        print(f"❌ Папка {PDF_DIR} не существует!")
        return False

    pdf_files = glob.glob(os.path.join(PDF_DIR, "*.pdf"))
    if not pdf_files:
        print("⚠️ В папке data нет PDF файлов")
        return False

    print(f"📄 Найдено PDF: {len(pdf_files)}")

    converter = PdfConverter(artifact_dict=create_model_dict())

    embed_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cuda 'if torch.cuda.is_available() else 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=embed_model
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=300,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )

    for pdf_path in pdf_files:
        file_name = os.path.basename(pdf_path)
        print(f"\n🔄 Обрабатываю: {file_name}")

        try:
            rendered = converter(pdf_path)
            text, _, _ = text_from_rendered(rendered)
        except Exception as e:
            print(f"❌ Ошибка Marker при обработке {file_name}: {e}")
            continue

        chunks = splitter.split_text(text)
        print(f"📦 Создано чанков: {len(chunks)}")

        documents = [
            Document(
                page_content=chunk,
                metadata={
                    "source": file_name,
                    "chunk_id": i
                }
            )
            for i, chunk in enumerate(chunks)
        ]
        
        vector_store.add_documents(documents)
        print(f"✅ Добавлено в векторную БД: {len(documents)} документов")

    print("\nPDF обработаны и записаны в Chroma.")
    return True


async def aquery_resp(question: str, k: int = 3):
    """
    Args:
        question: Вопрос пользователя
        k: Количество документов для поиска (по умолчанию 3)
    
    Returns:
        Ответ от GigaChat на основе найденного контекста
    """
    embed_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embed_model,
        persist_directory=CHROMA_DIR
    )

    llm = GigaChat(
        credentials=GIGACHAT_KEY,
        model="GigaChat-Pro",  
        verify_ssl_certs=False,
        temperature=0.2,
        max_tokens=2000
    )

    prompt = ChatPromptTemplate.from_template("""
        Ты — интеллектуальный помощник сервиса МОСБИРЖА. 
        Твоя задача — отвечать на вопросы пользователей, используя только предоставленный контекст из документации.

        Правила:
        1. Отвечай строго на основе контекста ниже
        2. Если информации нет в контексте — скажи: "Информация не найдена. Обратитесь в поддержку @support_mosbirzha"
        3. Отвечай кратко, структурированно и по существу
        4. Используй форматирование для удобства чтения
        5. Если нужно, ссылайся на источник из метаданных

        Контекст из документов:
        {context}

        Вопрос пользователя: {question}

        Ответ:
    """)

    retrieved_docs = vector_store.similarity_search(question, k=k)
    
    if not retrieved_docs:
        return "⚠️ Релевантные документы не найдены. Обратитесь в поддержку @support_service"

    context_parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        source = doc.metadata.get('source', 'Неизвестный источник')
        context_parts.append(f"[Документ {i}: {source}]\n{doc.page_content}")
    
    docs_content = "\n\n---\n\n".join(context_parts)

    try:
        message = await prompt.ainvoke({
            'question': question,
            'context': docs_content
        })
        
        answer = await llm.ainvoke(message)
        return answer.content
    
    except Exception as e:
        print(f"❌ Ошибка при запросе к GigaChat: {e}")
        return "Произошла ошибка при обработке запроса. Попробуйте позже или обратитесь в поддержку."


def query_resp_sync(question: str, k: int = 3):
    return asyncio.run(aquery_resp(question, k))


if __name__ == '__main__':
    success = True
    
    if success:
        print("\n" + "="*60)
        print("🎉 Готово! Теперь можно задавать вопросы.")
        print("="*60)
        
        test_question =  "Кем утверждаются правила платформы?"
        answer = query_resp_sync(test_question)
        print(f"\n🤖 Ответ: {answer}")