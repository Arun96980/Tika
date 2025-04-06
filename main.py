import os
import json
import requests
import spacy
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def extract_clean_text(file_path, tika_url="http://localhost:9998/rmeta/text"):
    try:
        with open(file_path, "rb") as f:
            headers = {"Accept": "application/json"}
            response = requests.put(tika_url, data=f, headers=headers)
            response.raise_for_status()
            content_json = response.json()
            if content_json and isinstance(content_json, list):
                return content_json[0].get("X-TIKA:content", "").strip()
    except Exception as e:
        print(f"❌ Error extracting from {file_path}: {e}")
    return ""

def clean_sentence(sentence):
    return sentence.replace('\n\n', ' ').replace('\n', ' ').replace('', '').strip()

def split_into_sentences(text, nlp):
    doc = nlp(text)
    return [clean_sentence(sent.text) for sent in doc.sents if sent.text.strip() and len(sent.text.strip()) > 5]

def process_resumes(pdf_dir, model, index_file="faiss.index", data_file="resume_data.json", tika_url="http://localhost:9998/rmeta/text"):
    results = []
    vectors = []
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(('.pdf', '.docx', '.pptx', '.txt'))]

    if not pdf_files:
        print("📂 No supported files found.")
        return

    nlp = spacy.load("en_core_web_sm")

    for pdf_file in tqdm(pdf_files, desc="🔍 Processing Files"):
        file_path = os.path.join(pdf_dir, pdf_file)
        text = extract_clean_text(file_path, tika_url)
        if not text:
            continue
        sentences = split_into_sentences(text, nlp)
        for sentence in sentences:
            input_text = f"passage: {sentence}"
            embedding = model.encode(input_text, normalize_embeddings=True)
            vectors.append(embedding)
            results.append({
                "file": pdf_file,
                "sentence": sentence,
                "length": len(sentence)
            })

    if results:
        dim = vectors[0].shape[0]
        index = faiss.IndexFlatIP(dim)
        index.add(np.array(vectors).astype("float32"))

        faiss.write_index(index, index_file)
        with open(data_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Indexed {len(results)} sentences. Saved FAISS index and metadata.")

def search(query, model, index_file="faiss.index", data_file="resume_data.json", top_k=5):
    query_vector = model.encode(f"query: {query}", normalize_embeddings=True).astype("float32")
    index = faiss.read_index(index_file)

    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    scores, indices = index.search(np.array([query_vector]), top_k)
    print(f"\n🔍 Top {top_k} results for query: '{query}'\n")
    for i, idx in enumerate(indices[0]):
        print(f"{i+1}. {data[idx]['sentence']} (File: {data[idx]['file']})")

if __name__ == "__main__":
    model = SentenceTransformer("intfloat/e5-large")

    pdf_directory = "resumes"
    process_resumes(pdf_directory, model)

    while True:
        query = input("\n🔎 Enter a search query (or 'exit' to quit): ")
        if query.lower() == "exit":
            break
        search(query, model)
