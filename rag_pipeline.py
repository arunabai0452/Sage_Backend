import os
import json
import asyncio
import sys
import re
from typing import List, TypedDict
from urllib.parse import urlparse

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.cache import InMemoryCache
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langgraph.graph import START, StateGraph
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.language_models.chat_models import BaseChatModel

import numpy as np
from collections import Counter

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import langchain
langchain.llm_cache = InMemoryCache()

llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.7, max_tokens=500)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)

recent_refs = [
    "What is the most recent publication?",
    "Recent research papers?",
    "Latest contributions?",
    "New work?",
    "Work from 2022 or 2023?"
]
ref_vectors = [embeddings.embed_query(q) for q in recent_refs]
avg_recent_reference = np.mean(ref_vectors, axis=0)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_most_mentioned_person_name(docs: List[Document]) -> str:
    names = []
    for doc in docs:
        matches = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", doc.page_content)
        names.extend(matches)
    if not names:
        return ""
    most_common = Counter(names).most_common(1)
    return most_common[0][0] if most_common else ""

def flatten_json(json_obj, parent_key='', sep=' > '):
    items = []
    if isinstance(json_obj, dict):
        for k, v in json_obj.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            items.extend(flatten_json(v, new_key, sep=sep))
    elif isinstance(json_obj, list):
        for i, item in enumerate(json_obj):
            items.extend(flatten_json(item, f"{parent_key}[{i}]", sep=sep))
    else:
        value = str(json_obj).strip()
        if "\n" in value:
            lines = value.split("\n")
            for i, line in enumerate(lines):
                items.append((f"{parent_key}[{i}]", line.strip()))
        else:
            items.append((parent_key, value))
    return items

def load_json_data(file_path):
    if not os.path.exists(file_path):
        print(f"❌ File '{file_path}' not found.")
        return []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            flattened_data = flatten_json(data)
            content = "\n".join([f"{key}: {value}" for key, value in flattened_data])
            data = [{
                "title": f"{data.get('name', 'Unknown')} - {data.get('designation', 'Unknown')}",
                "content": content
            }]
        if not isinstance(data, list):
            raise ValueError("Expected JSON to be a list of dictionaries.")
        return data
    except json.JSONDecodeError:
        print("❌ Error: JSON file is not properly formatted.")
        return []
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return []

def extract_year_for_metadata(text: str) -> int:
    matches = re.findall(r"\b(20\d{2}|19\d{2})\b", text)
    years = [int(y) for y in matches if 1900 <= int(y) <= 2100]
    return max(years) if years else 0

def create_faiss_index():
    data = load_json_data("data.json")
    if not data:
        print("⚠️ No valid JSON data found.")
        return None
    doc_objects = []
    for item in data:
        content = item["content"]
        year = extract_year_for_metadata(content)
        title = item.get("title", "Unknown")
        doc_objects.append(Document(
            page_content=content,
            metadata={"title": title, "year": year}
        ))
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = text_splitter.split_documents(doc_objects)
    return FAISS.from_documents(documents=split_docs, embedding=embeddings)

vector_store = create_faiss_index()
if not vector_store:
    print("❌ Vector store could not be created. Exiting.")
    sys.exit(1)

class State(TypedDict):
    question: str
    context: List[Document]
    email: str
    answer: str

pronoun_to_name = {}

def update_pronoun_to_name(pronoun, name):
    if pronoun and name:
        pronoun_to_name[pronoun] = name

def get_name_from_pronoun(pronoun):
    return pronoun_to_name.get(pronoun)

def extract_name_from_question(question: str) -> str:
    match = re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", question)
    return match.group(1).strip() if match else ""

def detect_gender_from_text(text: str) -> str:
    text = f" {text.lower()} "
    if " she " in text or " her " in text:
        return "she"
    elif " he " in text or " his " in text:
        return "he"
    elif " they " in text or " their " in text:
        return "they"
    else:
        return ""

def extract_email(text):
    match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    return match.group(0) if match else None

def extract_years_from_question(q: str) -> List[int]:
    return [int(y) for y in re.findall(r"\b(20\d{2}|19\d{2})\b", q)]

def retrieve(state: State):
    question = state["question"]
    question_embedding = embeddings.embed_query(question)
    recent_intent_score = cosine_similarity(question_embedding, avg_recent_reference)
    is_recent = recent_intent_score > 0.6

    target_name = extract_name_from_question(question)
    pronoun_in_question = next((p for p in ["he", "she", "his", "her", "they", "their"] if f" {p} " in question.lower()), None)
    if not target_name and pronoun_in_question:
        target_name = "Anuj Sharma"

    target_years = extract_years_from_question(question)
    retrieved_docs = vector_store.similarity_search(question, k=15)

    person_docs = [
        doc for doc in retrieved_docs
        if target_name.lower() in doc.page_content.lower()
    ]
    retrieved_docs = person_docs if person_docs else retrieved_docs

    filtered_docs = [doc for doc in retrieved_docs if doc.metadata.get("year", 0) in target_years]

    if target_years and filtered_docs:
        selected_docs = filtered_docs[:5]
    elif is_recent:
        sorted_docs = sorted(retrieved_docs, key=lambda d: d.metadata.get("year", 0), reverse=True)
        selected_docs = sorted_docs[:3]
    else:
        selected_docs = retrieved_docs[:5]

    # After selected_docs is created:
    selected_text = " ".join(doc.page_content for doc in selected_docs)

    # 🔍 Try inferring the most mentioned name from selected docs
    inferred_name = get_most_mentioned_person_name(selected_docs)
    if inferred_name:
        target_name = inferred_name  # override any fallback or partial match

    # Store inferred pronoun → person name mapping
    gender = detect_gender_from_text(selected_text)
    update_pronoun_to_name("his", target_name)
    update_pronoun_to_name("her", target_name)
    update_pronoun_to_name("their", target_name)
    emails = [extract_email(doc.page_content) for doc in selected_docs if extract_email(doc.page_content)]
    contact_email = emails[0] if emails else None

    return {"context": selected_docs, "email": contact_email}

def format_answer_text(text):
    # ✅ Convert Markdown-style bold **Label** to <b> Label </b>
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)

    # ✅ Convert [label](url) to proper <a> tag
    text = re.sub(
        r"\[([^\[\]]+?)\]\((https?://[^\s)]+)\)",
        r'<a href="\2" target="_blank">\1</a>',
        text
    )

    # ✅ Remove nested malformed cases like: [Label](<a href="URL">URL</a>)
    text = re.sub(
        r"\[([^\[]+?)\]\(<a href=\"(https?://[^\"]+)\"[^>]*>\2</a>\)",
        r'<a href="\2" target="_blank">\1</a>',
        text
    )

    # ✅ Detect raw links and label them nicely
    def label_known_links(match):
        url = match.group(0)
        domain = urlparse(url).netloc.replace("www.", "")
        label = domain.replace(".com", "").replace(".org", "").replace(".edu", "")
        label = label.replace("scholar.google", "Google Scholar")
        label = re.sub(r'[-_]', ' ', label).title()
        return f'<a href="{url}" target="_blank">{label}</a>'

    # Replace plain raw links (without markdown) with labeled links
    text = re.sub(r"(?<!href=\")https?://[^\s<>\"]+", label_known_links, text)

    return text


async def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    question = state["question"].strip()
    email = state.get("email")
    extracted_years = extract_years_from_question(question)

    # Before building prompt
    pronoun = next((p for p in ["he", "she", "they", "his", "her", "their"] if f" {p} " in question.lower()), None)
    person_name = get_name_from_pronoun(pronoun) if pronoun else None
    if pronoun and person_name:
        pattern = re.compile(rf"\b{pronoun}\b", flags=re.IGNORECASE)
        question = pattern.sub(person_name, question)

    if not docs_content.strip():
        return {
            "answer": (
                "❌ Sorry, no relevant information was found.\n"
                + (f"📧 Contact: <a href='mailto:{email}'>{email}</a>" if email else "📧 Email support@example.com.")
            )
        }

    if extracted_years:
        prompt = f"Use only documents from {extracted_years[0]}.\n\nContext:\n{docs_content}\n\nQuestion: {question}"
    elif any(word in question.lower() for word in ["recent", "latest"]):
        prompt = f"Use the most recent documents (2023 preferred).\n\nContext:\n{docs_content}\n\nQuestion: {question}"
    else:
        prompt = f"Context:\n{docs_content}\n\nQuestion: {question}"

    response = await llm.ainvoke(prompt)
    answer = format_answer_text(response.content.strip())
    return {"answer": answer}

graph_builder = StateGraph(State)
graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("generate", generate)
graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "generate")
graph = graph_builder.compile()

async def main():
    print("\n💡 Ask your question (type 'exit' to quit):\n")
    try:
        while True:
            question = input("🔹 Question: ").strip()
            if question.lower() == "exit":
                print("👋 Exiting... Have a great day!")
                break
            response = await graph.ainvoke({"question": question})
            print(f"\n✅ Answer:\n{response['answer']}\n{'-'*60}\n")
    except KeyboardInterrupt:
        print("\n👋 Exiting... Have a great day!")

if __name__ == "__main__":
    asyncio.run(main())