from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import uuid


embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://localhost:11435"
        )

vectorstore = Chroma(embedding_function=embeddings)


def create_doctor_documents():
    """Создание документов с информацией о врачах больницы"""

    doctors_data = [
        {
            "doctor_name": "Иванова Анна Петровна",
            "cabinet_number": "101",
            "floor": "1",
            "specialization": "Терапевт",
            "description": "Врач-терапевт высшей категории. Стаж работы 15 лет. Принимает взрослых пациентов с общими заболеваниями."
        },
        {
            "doctor_name": "Смирнов Дмитрий Васильевич",
            "cabinet_number": "205",
            "floor": "2",
            "specialization": "Хирург",
            "description": "Хирург-ортопед. Специализируется на операциях на позвоночнике и суставах. Стаж 20 лет."
        },
        {
            "doctor_name": "Кузнецова Елена Сергеевна",
            "cabinet_number": "312",
            "floor": "3",
            "specialization": "Кардиолог",
            "description": "Врач-кардиолог, кандидат медицинских наук. Диагностика и лечение сердечно-сосудистых заболеваний."
        },
        {
            "doctor_name": "Петров Алексей Николаевич",
            "cabinet_number": "408",
            "floor": "4",
            "specialization": "Невролог",
            "description": "Невролог, специалист по лечению мигреней и заболеваний периферической нервной системы."
        },
        {
            "doctor_name": "Соколова Мария Игоревна",
            "cabinet_number": "105",
            "floor": "1",
            "specialization": "Педиатр",
            "description": "Детский врач. Принимает детей от 0 до 18 лет. Вакцинация, диспансеризация."
        },
        {
            "doctor_name": "Васильев Игорь Борисович",
            "cabinet_number": "209",
            "floor": "2",
            "specialization": "Отоларинголог",
            "description": "ЛОР-врач. Лечение заболеваний уха, горла и носа. Проводит эндоскопические операции."
        },
        {
            "doctor_name": "Николаева Ольга Владимировна",
            "cabinet_number": "315",
            "floor": "3",
            "specialization": "Офтальмолог",
            "description": "Врач-офтальмолог. Диагностика и лечение заболеваний глаз. Подбор очков и линз."
        },
        {
            "doctor_name": "Алексеев Сергей Михайлович",
            "cabinet_number": "417",
            "floor": "4",
            "specialization": "Гастроэнтеролог",
            "description": "Специалист по заболеваниям желудочно-кишечного тракта. Проводит гастроскопию."
        },
        {
            "doctor_name": "Морозова Татьяна Александровна",
            "cabinet_number": "110",
            "floor": "1",
            "specialization": "Гинеколог",
            "description": "Врач-гинеколог. Ведение беременности, лечение гинекологических заболеваний."
        },
        {
            "doctor_name": "Федоров Павел Дмитриевич",
            "cabinet_number": "218",
            "floor": "2",
            "specialization": "Уролог",
            "description": "Врач-уролог. Диагностика и лечение заболеваний мочеполовой системы у мужчин и женщин."
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


def add_documents(documents):
    vectorstore.add_documents(documents)


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

def similarity_search(query):
    return vectorstore.similarity_search(query)


def delete_by_id(document_id):
    vectorstore.delete(ids=[document_id])

def update_document(
    doc_id: str,
    new_text: str

):
    vectorstore.delete(ids=[doc_id])

    doc = Document(
        page_content=new_text

    )

    vectorstore.add_documents(
        documents=[doc],
        ids=[doc_id]
    )




if __name__ == "__main__":
    texts = [
        Document(page_content="Python is snake"),
        Document(page_content="JavaScript is not java")
    ]

    # add_documents_to_db(texts)

    # similar_docs = similarity_search('')
    # print(similar_docs)
    #
    # first_doc_id = list(vectorstore._collection.get()['ids'])[0]
    # delete_by_id(first_doc_id)

    # add_documents_to_db(create_doctor_documents())

