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
                main_text = content_json[0].get("X-TIKA:content", "")
                return main_text.strip()
    except Exception as e:
        print(f"❌ Error extracting from {file_path}: {e}")
    return ""

def clean_sentence(text):
    return (
        text.replace("", "")
            .replace("•", "")
            .replace("\n", " ")
            .replace("\t", " ")
            .replace("  ", " ")
            .strip()
    )

def split_into_sentences(text, nlp):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip() and len(sent.text.strip()) > 5]

def process_and_index(pdf_dir, output_json, faiss_index_path, tika_url="http://localhost:9998/rmeta/text"):
    results = []
    supported_exts = ('.pdf', '.doc', '.docx', '.pptx', '.txt')
    input_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(supported_exts)]

    if not input_files:
        print(f"📂 No supported files found in {pdf_dir}")
        return [], None

    # Load NLP & model
    nlp = spacy.load("en_core_web_sm")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    dim = 384
    index = faiss.IndexFlatL2(dim)
    metadata = []

    for input_file in tqdm(input_files, desc="🔍 Processing Files"):
        file_path = os.path.join(pdf_dir, input_file)
        text = extract_clean_text(file_path, tika_url)
        if not text:
            continue

        sentences = split_into_sentences(text, nlp)
        for sentence in sentences:
            cleaned = clean_sentence(sentence)
            if cleaned:
                embedding = model.encode(cleaned)
                index.add(np.array([embedding]).astype("float32"))
                metadata.append({
                    "file": input_file,
                    "sentence": cleaned
                })
                results.append({
                    "file": input_file,
                    "sentence": cleaned,
                    "length": len(cleaned)
                })

    # Save index and metadata
    faiss.write_index(index, faiss_index_path)
    with open("faiss_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    with open(output_json, "w", encoding="utf-8") as out_f:
        json.dump(results, out_f, indent=2, ensure_ascii=False)

    print(f"\n✅ Indexed {len(results)} sentences. Saved metadata and FAISS index.")
    return metadata, model

def query_loop(model, faiss_index_path, metadata_path):
    print("\n🔍 Ready for querying! Type your query below (or type 'exit' to quit):")
    index = faiss.read_index(faiss_index_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    while True:
        query = input("\n🔎 Enter query: ").strip()
        if query.lower() in {"exit", "quit"}:
            print("👋 Exiting search.")
            break

        embedding = model.encode(query).astype("float32").reshape(1, -1)
        D, I = index.search(embedding, k=5)

        print("\n🔗 Top 5 matches:")
        for i, idx in enumerate(I[0]):
            print(f"{i+1}. {metadata[idx]['sentence']} \n   ↪ from file: {metadata[idx]['file']}\n")

if __name__ == "__main__":
    pdf_directory = "resumes"                      # Folder with your resumes
    output_file = "resumes_main_content.json"      # Sentences JSON
    faiss_index_file = "resumes_index.faiss"       # FAISS index file
    tika_server_url = "http://localhost:9998/rmeta/text"

    # Step 1: Process and build index
    metadata, model = process_and_index(
        pdf_directory, output_file, faiss_index_file, tika_url=tika_server_url
    )

    # Step 2: Query loop if successful
    if metadata and model:
        query_loop(model, faiss_index_file, "faiss_metadata.json")
