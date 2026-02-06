import asyncio
import json
from typing import TypedDict, Dict, Any, List
from langchain_chroma import Chroma
import httpx
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings

import uuid
llm = ChatOllama(
            model="tinyllama:1.1b",  # или "codellama:7b"
            temperature=0.1,
            base_url="http://localhost:11434"  # порт LLM контейнера
        )

embeddings = OllamaEmbeddings(
            model="nomic-embed-text",  # загруженная в ollama_embeddings
            base_url="http://localhost:11435"  # порт эмбеддинг контейнера!
        )
vectorstore = Chroma(
                persist_directory="./hospital",
                embedding_function=embeddings
                )

def create_doctor_documents():
    """Создание документов с информацией о врачах больницы"""

    doctors_data = [
{
"doctor_name": "Ivanova Anna Petrovna",
"cabinet_number": "101",
"floor": "1",
"specialization": "Therapist",
            "description": "A general practitioner of the highest category. Work experience of 15 years. Accepts adult patients with common diseases."
        },
        {
"doctor_name": "Dmitry Vasilyevich Smirnov",
            "cabinet_number": "205",
"floor": "2",
"specialization": "Surgeon",
            "description": "Orthopedic surgeon. Specializes in spine and joint surgery. Experience of 20 years."
        },
        {
"doctor_name": "Kuznetsova Elena Sergeevna",
"cabinet_number": "312",
"floor": "3",
"specialization": "Therapist",
"description": "Cardiologist, Candidate of Medical Sciences. Diagnosis and treatment of cardiovascular diseases."
},
        {
"doctor_name": "Petrov Alexey Nikolaevich",
            "cabinet_number": "408",
            "floor": "4",
            "specialization": "Neurologist",
"description": "Neurologist, specialist in the treatment of migraines and diseases of the peripheral nervous system."
        },
        {
"doctor_name": "Sokolova Maria Igorevna",
"cabinet_number": "105",
"floor": "1",
"specialization": "Pediatrician",
"description": "Pediatrician. Accepts children from 0 to 18 years old. Vaccination, medical examination."
},
        {
"doctor_name": "Vasiliev Igor Borisovich",
            "cabinet_number": "209",
            "floor": "2",
            "specialization": "Otolaryngologist",
"description": "ENT doctor. Treatment of diseases of the ear, throat and nose. Performs endoscopic surgeries."
},
        {
"doctor_name": "Nikolaeva Olga Vladimirovna",
"cabinet_number": "315",
"floor": "3",
"specialization": "Ophthalmologist",
"description": "Ophthalmologist. Diagnosis and treatment of eye diseases. Selection of glasses and lenses."
        },
        {
"doctor_name": "Alekseev Sergey Mikhailovich",
            "cabinet_number": "417",
            "floor": "4",
"specialization": "Gastroenterologist",
"description": "Specialist in diseases of the gastrointestinal tract. Performs a gastroscopy."
        },
        {
"doctor_name": "Tatiana Alexandrovna Morozova",
"cabinet_number": "110",
"floor": "1",
"specialization": "Gynecologist",
"description": "Gynecologist. Pregnancy management, treatment of gynecological diseases."
},
        {
"doctor_name": "Pavel Dmitrievich Fedorov",
            "cabinet_number": "218",
"floor": "2",
"specialization": "Urologist",
"description": "Urologist. Diagnosis and treatment of diseases of the genitourinary system in men and women."
}
    ]
    documents = []

    for doctor in doctors_data:
        # Создаем текстовое представление врача для поиска
        doctor_text = (
            f"Врач: {doctor['doctor_name']}. "
            f"Специализация: {doctor['specialization']}. "
            f"Кабинет: {doctor['cabinet_number']}. "
            f"Этаж: {doctor['floor']}. "
            f"Описание: {doctor['description']}"
        )

        # Создаем метаданные с полной информацией
        metadata = {
            "doctor_name": doctor["doctor_name"],
            "cabinet_number": doctor["cabinet_number"],
            "floor": doctor["floor"],
            "specialization": doctor["specialization"],
            "description": doctor["description"],
            "type": "doctor",
            "source": "hospital_database"
        }

        # Создаем документ
        doc = Document(
            page_content=doctor_text,
            metadata=metadata
        )

        documents.append(doc)

    return documents

def get_all_doctors():
    """Получение всех врачей из базы"""
    # Получаем все документы
    results = vectorstore.get()

    doctors = []
    for i, (content, metadata) in enumerate(zip(results['documents'], results['metadatas'])):
        doctors.append({
            "id": results['ids'][i],
            "content": content,
            **metadata
        })

    return doctors


def find_doctor_by_specialization(specialization: str):
    """Поиск врачей по специализации"""
    results = vectorstore.similarity_search(
        f"специализация {specialization}",
        k=10
    )

    doctors = []
    for doc in results:
        if doc.metadata['specialization'].lower() == specialization.lower():
            doctors.append({
                "name": doc.metadata['doctor_name'],
                "cabinet": doc.metadata['cabinet_number'],
                "floor": doc.metadata['floor']
            })

    return doctors


def find_doctor_by_name(name: str):
    """Поиск врача по имени"""
    results = vectorstore.similarity_search(
        f"врач {name}",
        k=5
    )

    for doc in results:
        if name.lower() in doc.metadata['doctor_name'].lower():
            return {
                "name": doc.metadata['doctor_name'],
                "specialization": doc.metadata['specialization'],
                "cabinet": doc.metadata['cabinet_number'],
                "floor": doc.metadata['floor'],
                "description": doc.metadata['description']
            }

    return None


def update_doctor_info(doctor_id: str, new_info: dict):
    """Обновление информации о враче"""
    # Получаем текущий документ
    current_doc = vectorstore.get(ids=[doctor_id])

    if not current_doc['documents']:
        print(f"Врач с ID {doctor_id} не найден")
        return False

    # Обновляем метаданные
    updated_metadata = {**current_doc['metadatas'][0], **new_info}

    # Обновляем текст документа
    updated_text = (
        f"Врач: {updated_metadata['doctor_name']}. "
        f"Специализация: {updated_metadata['specialization']}. "
        f"Кабинет: {updated_metadata['cabinet_number']}. "
        f"Этаж: {updated_metadata['floor']}. "
        f"Описание: {updated_metadata['description']}"
    )

    # Обновляем документ в базе
    vectorstore.update_document(
        document_id=doctor_id,
        document=Document(
            page_content=updated_text,
            metadata=updated_metadata
        )
    )

    vectorstore.persist()
    print(f"Информация о враче {updated_metadata['doctor_name']} обновлена")
    return True


# Пример использования дополнительных функций
def test_functions():
    # Получить всех врачей
    all_doctors = get_all_doctors()
    print(f"Всего врачей в базе: {len(all_doctors)}")

    # Найти всех хирургов
    surgeons = find_doctor_by_specialization("хирург")
    print(f"\nХирурги в больнице: {len(surgeons)}")
    for surgeon in surgeons:
        print(f"  - {surgeon['name']}, каб. {surgeon['cabinet']}")

    # Найти конкретного врача
    doctor = find_doctor_by_name("Иванова")
    if doctor:
        print(f"\nНайден врач: {doctor['name']}")
        print(f"  Специализация: {doctor['specialization']}")
        print(f"  Кабинет: {doctor['cabinet']} (этаж {doctor['floor']})")

def add_documents_to_db(documents):
    """Добавление документов в векторную базу"""

    # Создаем уникальные ID для каждого документа
    ids = [f"doctor_{uuid.uuid4().hex[:8]}" for _ in range(len(documents))]

    # Добавляем документы в векторную базу
    vectorstore.add_documents(
        documents=documents,
        ids=ids
    )


    print(f"Успешно добавлено {len(documents)} документов о врачах")

    return ids

def build_context(docs) -> str:
    # f"Специализация: {updated_metadata['specialization']}. "
    # f"Кабинет: {updated_metadata['cabinet_number']}. "
    return "/n/n".join(f"{doc}" for doc in docs)



#cделать тоже самое на руками
# documents = create_doctor_documents()
# add_documents_to_db(documents)
# print(get_all_doctors())
# print(find_doctor_by_specialization("Педиатр"))
doctors = find_doctor_by_specialization("Therapist")
print(doctors)
context = build_context(doctors)

print(context)
question = "what is Therapist's cabinet numbers?"
template=f"You are a nurse working at the registry office, who answers patients' questions. All the background information is here: {context} Patient's question: {question} Your answer is:"
print(llm.invoke(template))

#починить find_doctor_by_specialization
#еще промптов добавить с выбором








