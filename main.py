import os
import json
import requests
import spacy
import faiss
import numpy as np
import hashlib
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor

# -----------------------------
# Helper: Compute MD5 hash of a file
def compute_file_hash(file_path):
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
    except Exception as e:
        print(f"Error computing hash for {file_path}: {e}")
    return hash_md5.hexdigest()

# -----------------------------
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

# -----------------------------
def clean_sentence(sentence):
    return sentence.replace('\n\n', ' ').replace('\n', ' ').replace('', '').strip()

# -----------------------------
def split_into_sentences(text, nlp):
    doc = nlp(text)
    return [clean_sentence(sent.text) for sent in doc.sents if sent.text.strip() and len(sent.text.strip()) > 5]

# -----------------------------
def process_file(file, pdf_dir, nlp, model, tika_url):
    """
    Process a single file:
      - Compute its hash.
      - Extract text and split into sentences.
      - For each sentence, compute the embedding.
    Returns a list of dictionaries (one per sentence) with file info and embedding.
    """
    file_path = os.path.join(pdf_dir, file)
    file_hash = compute_file_hash(file_path)
    text = extract_clean_text(file_path, tika_url)
    if not text:
        return []
    sentences = split_into_sentences(text, nlp)
    results = []
    for sentence in sentences:
        if sentence:
            embedding = model.encode(f"passage: {sentence}", normalize_embeddings=True)
            results.append({
                "file": file,
                "file_hash": file_hash,
                "sentence": sentence,
                "length": len(sentence),
                "embedding": embedding.tolist()  # temporary, for caching; not saved in final metadata
            })
    return results

# -----------------------------
def load_existing_metadata(metadata_file):
    if os.path.exists(metadata_file):
        with open(metadata_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

# -----------------------------
def process_all_files(pdf_dir, model, tika_url="http://localhost:9998/rmeta/text", metadata_file="faiss_metadata.json"):
    """
    Process files concurrently and do incremental updates.
    Loads existing metadata and builds a set of processed file hashes.
    Only processes files that are new or updated.
    Returns a list of new sentence entries (including embedding vectors).
    """
    supported_exts = ('.pdf', '.doc', '.docx', '.pptx', '.txt')
    files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(supported_exts)]
    
    if not files:
        print("📂 No supported files found.")
        return []
    
    # Load existing metadata and get processed file hashes
    existing_metadata = load_existing_metadata(metadata_file)
    processed_hashes = {entry["file_hash"] for entry in existing_metadata}
    
    nlp = spacy.load("en_core_web_sm")
    new_results = []
    
    # Process files in parallel
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, file, pdf_dir, nlp, model, tika_url) for file in files]
        for future in tqdm(futures, desc="🔁 Processing Files"):
            file_results = future.result()
            # Check the file hash in the first entry (if any)
            if file_results:
                file_hash = file_results[0].get("file_hash")
                if file_hash in processed_hashes:
                    # Skip this file if already processed.
                    continue
                new_results.extend(file_results)
    return new_results

# -----------------------------
def build_incremental_index(new_results, model, index_file="faiss.index", metadata_file="faiss_metadata.json"):
    """
    Build or update the FAISS index with new_results.
    new_results should have an "embedding" key containing the embedding vector.
    Existing metadata is loaded and new_results are appended.
    The embeddings from new_results are added to the FAISS index.
    """
    # Load existing metadata if available.
    existing_metadata = load_existing_metadata(metadata_file)
    all_metadata = existing_metadata.copy()
    
    # Determine embedding dimension using model on a sample text.
    sample_vec = model.encode("example", normalize_embeddings=True)
    dim = sample_vec.shape[0]
    
    # If index file exists, load it; otherwise create a new one.
    if os.path.exists(index_file):
        index = faiss.read_index(index_file)
    else:
        index = faiss.IndexFlatL2(dim)
    
    new_vectors = []
    for item in new_results:
        vec = np.array(item["embedding"]).astype("float32")
        new_vectors.append(vec)
        # Remove embedding from metadata before saving.
        item.pop("embedding", None)
        all_metadata.append(item)
    
    if new_vectors:
        new_vectors = np.array(new_vectors)
        index.add(new_vectors)
        faiss.write_index(index, index_file)
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(all_metadata, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Indexed {len(new_vectors)} new sentences. Updated FAISS index and metadata.")
    else:
        print("\n✅ No new files to process. Index remains unchanged.")

# -----------------------------
def search(query, model, index_file="faiss.index", metadata_file="faiss_metadata.json", top_k=5):
    query_vec = model.encode(f"query: {query}", normalize_embeddings=True).astype("float32")
    index = faiss.read_index(index_file)
    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    # Using L2 distance index (IndexFlatL2): lower distance means better match.
    distances, indices = index.search(np.array([query_vec]), top_k)
    print(f"\n🔍 Top {top_k} results for query: '{query}'\n")
    for i, idx in enumerate(indices[0]):
        print(f"{i+1}. {metadata[idx]['sentence']} (File: {metadata[idx]['file']}), Distance: {distances[0][i]:.3f}")

# -----------------------------
if __name__ == "__main__":
    model = SentenceTransformer("intfloat/e5-large")
    
    pdf_directory = "resumes"                      # Folder with your documents
    tika_url = "http://localhost:9998/rmeta/text"  # Tika endpoint
    index_file = "faiss.index"                     # FAISS index file
    metadata_file = "faiss_metadata.json"          # Metadata mapping file

    # Process files and get new sentence results (skip already processed files)
    new_results = process_all_files(pdf_directory, model, tika_url, metadata_file)
    
    # Build (or update) the FAISS index incrementally
    build_incremental_index(new_results, model, index_file, metadata_file)
    
    # Query loop:
    while True:
        query = input("\n🔎 Enter a search query (or 'exit' to quit): ").strip()
        if query.lower() == "exit":
            break
        search(query, model, index_file, metadata_file, top_k=5)
