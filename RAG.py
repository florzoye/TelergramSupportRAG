import os
import glob
from config import OPENAI_KEY

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


PDF_DIR = "data"
CHROMA_DIR = "data/chroma_db"
COLLECTION_NAME = "rules"


def process_pdfs_marker():

    if not os.path.exists(PDF_DIR):
        print(f" Папка {PDF_DIR} не существует!")
        return False

    pdf_files = glob.glob(os.path.join(PDF_DIR, "*.pdf"))
    if not pdf_files:
        print("В папке data нет PDF файлов")
        return False

    print(f" Найдено PDF: {len(pdf_files)}")

    converter = PdfConverter(artifact_dict=create_model_dict())

    embed_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=embed_model
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=400,
        separators=["\n\n", "\n", " ", ""]
    )

    for pdf_path in pdf_files:
        file_name = os.path.basename(pdf_path)
        print(f"\n Обрабатываю: {file_name}")

        try:
            rendered = converter(pdf_path)
            text, _, _ = text_from_rendered(rendered )
        except Exception as e:
            print(f"Ошибка Marker при обработке {file_name}: {e}")
            continue

        txt_path = os.path.join(PDF_DIR, f"{file_name}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)

        print(f" Результат распознавания сохранён: {txt_path}")

        chunks = splitter.split_text(text)
        print(f" Чанков: {len(chunks)}")

        for ch in chunks:
            doc = {
                "page_content": f"Файл: {file_name}\n{ch}",
                "metadata": {"source": file_name}
            }
            vector_store.add_documents([doc])

    print("\n Все PDF обработаны и записаны в Chroma.")
    return True


async def aquery_resp(question: str):
    embed_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    vector_store = Chroma(
        collection_name='rules',
        embedding_function=embed_model,
        persist_directory='./data'
    )

    llm = ChatOpenAI(
        model="gpt-4o", 
        temperature=0.1,
        api_key=OPENAI_KEY
    )

    prompt = ChatPromptTemplate.from_template("""
        Ты системный помощник сервиса SERVICE_NAME. Отвечай только по теме.
        Если нет ответа — скажи: "Обратитесь в поддержку @support_service"

        Question: {question}
        Context: {context}
        Answer:
    """)

    retrived_docs = vector_store.similarity_search(question, k=2)
    docs_content = "\n".join([doc.page_content for doc in retrived_docs])

    message = await prompt.ainvoke({'question': question, "context": docs_content})
    
    answer = await llm.ainvoke(message)
    return answer.content

if __name__ == '__main__':
    process_pdfs_marker()