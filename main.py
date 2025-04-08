import os
import json
import argparse
import requests
import spacy
import faiss
import numpy as np
import hashlib
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor
from sklearn.feature_extraction.text import TfidfVectorizer

# -----------------------------
# Lazy load spaCy model
nlp_model = None
def get_nlp():
    global nlp_model
    if nlp_model is None:
        nlp_model = spacy.load("en_core_web_sm")
    return nlp_model

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
# Compute MD5 hash for a sentence to avoid re-embedding duplicates
def compute_sentence_hash(sentence):
    return hashlib.md5(sentence.encode("utf-8")).hexdigest()

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
def process_file(file, pdf_dir, model, tika_url, existing_sentence_hashes):
    """
    Process a single file:
      - Compute its hash.
      - Extract text and split into sentences.
      - Batch compute embeddings for new sentences.
    Returns a list of dictionaries with file info and embeddings.
    """
    file_path = os.path.join(pdf_dir, file)
    file_hash = compute_file_hash(file_path)
    text = extract_clean_text(file_path, tika_url)
    if not text:
        return []
    
    nlp = get_nlp()
    sentences = split_into_sentences(text, nlp)
    
    new_sentences = []
    results = []
    for sentence in sentences:
        sent_hash = compute_sentence_hash(sentence)
        if sent_hash in existing_sentence_hashes:
            continue
        new_sentences.append(sentence)
        results.append({
            "file": file,
            "file_hash": file_hash,
            "sentence": sentence,
            "length": len(sentence),
            "sentence_hash": sent_hash  # used for deduplication
        })
    
    # Batch embed new sentences
    if new_sentences:
        passages = [f"passage: {s}" for s in new_sentences]
        embeddings = model.encode(passages, normalize_embeddings=True)
        for idx, emb in enumerate(embeddings):
            results[idx]["embedding"] = emb.tolist()
    else:
        results = []
    
    return results

# -----------------------------
def load_existing_metadata(metadata_file):
    if os.path.exists(metadata_file):
        with open(metadata_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

# -----------------------------
def process_all_files(pdf_dir, model, tika_url, metadata_file):
    """
    Process files concurrently using threads and perform incremental indexing.
    Only processes new or updated sentences.
    Returns a list of new sentence entries (with embeddings).
    """
    supported_exts = ('.pdf', '.doc', '.docx', '.pptx', '.txt')
    files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(supported_exts)]
    
    if not files:
        print("📂 No supported files found.")
        return []
    
    existing_metadata = load_existing_metadata(metadata_file)
    processed_sentence_hashes = {entry.get("sentence_hash") for entry in existing_metadata if entry.get("sentence_hash")}
    
    new_results = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, file, pdf_dir, model, tika_url, processed_sentence_hashes) for file in files]
        for future in tqdm(futures, desc="🔁 Processing Files"):
            file_results = future.result()
            if file_results:
                new_results.extend(file_results)
    return new_results

# -----------------------------
def build_incremental_index(new_results, model, index_file, metadata_file):
    """
    Build or update the FAISS index with new_results.
    Loads existing metadata and appends new results.
    Embeddings from new_results are added to the FAISS index.
    """
    existing_metadata = load_existing_metadata(metadata_file)
    all_metadata = existing_metadata.copy()
    
    # Determine embedding dimension using a sample text.
    sample_vec = model.encode("example", normalize_embeddings=True)
    dim = sample_vec.shape[0]
    
    if os.path.exists(index_file):
        index = faiss.read_index(index_file)
    else:
        index = faiss.IndexFlatL2(dim)
    
    new_vectors = []
    for item in new_results:
        vec = np.array(item["embedding"]).astype("float32")
        new_vectors.append(vec)
        # Remove the embedding field from metadata before saving
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
        print("\n✅ No new sentences to process. Index remains unchanged.")

# -----------------------------
def update_metadata_with_tfidf(metadata):
    """
    Compute a TF-IDF weight for each sentence and update metadata.
    Here, the weight is computed as the sum of TF-IDF scores for all words in the sentence.
    """
    sentences = [entry["sentence"] for entry in metadata]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    tfidf_weights = tfidf_matrix.sum(axis=1).A1  # flatten to 1D array
    for entry, weight in zip(metadata, tfidf_weights):
        entry["tfidf_weight"] = weight
    return metadata

# -----------------------------
def aggregated_search_tfidf(query, model, index_file, metadata_file, top_k_sentences=100, top_k_resumes=10):
    """
    Aggregated search with normalization by total TF-IDF sum per resume.
    The similarity score is weighted by TF-IDF and normalized by the total TF-IDF sum
    of the contributing sentences for each resume.
    """
    try:
        # Encode the query
        query_vec = model.encode(f"query: {query}", normalize_embeddings=True).astype("float32")
        index = faiss.read_index(index_file)

        # Load and update metadata with TF-IDF weights
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        metadata = update_metadata_with_tfidf(metadata)

        # Retrieve top sentences
        distances, indices = index.search(np.array([query_vec]), top_k_sentences)

        # Aggregate weighted similarity scores by resume file
        resume_scores = {}
        tfidf_sums = {}

        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(metadata):
                continue
            item = metadata[idx]
            file = item["file"]
            base_similarity = 1 / (1 + dist)
            tfidf_weight = item.get("tfidf_weight", 1.0)
            weighted_similarity = base_similarity * tfidf_weight

            resume_scores[file] = resume_scores.get(file, 0.0) + weighted_similarity
            tfidf_sums[file] = tfidf_sums.get(file, 0.0) + tfidf_weight

        # Normalize aggregated scores by TF-IDF sum for each resume
        normalized_scores = {
            file: score / tfidf_sums[file]
            for file, score in resume_scores.items() if tfidf_sums[file] > 0
        }

        # Sort resumes by normalized score
        sorted_resumes = sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)

        print(f"\n🔍 Normalized Aggregated TF-IDF weighted search results for query: '{query}'\n")
        for rank, (file, score) in enumerate(sorted_resumes[:top_k_resumes], start=1):
            print(f"{rank}. Resume File: {file} - Normalized Score: {score:.3f}")
    except Exception as e:
        print(f"❌ Error during search: {e}")

# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Incremental FAISS-based Resume Search Engine with TF-IDF weighting")
    parser.add_argument("--pdf_dir", type=str, default="resumes", help="Directory containing resume files")
    parser.add_argument("--tika_url", type=str, default="http://localhost:9998/rmeta/text", help="Tika endpoint URL")
    parser.add_argument("--index_file", type=str, default="faiss.index", help="FAISS index file")
    parser.add_argument("--metadata_file", type=str, default="faiss_metadata.json", help="Metadata mapping file")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the FAISS index from scratch")
    args = parser.parse_args()
    
    # Optionally rebuild index and metadata files
    if args.rebuild:
        if os.path.exists(args.index_file):
            os.remove(args.index_file)
        if os.path.exists(args.metadata_file):
            os.remove(args.metadata_file)
        print("🔄 Rebuild selected: existing index and metadata removed.")
    
    # Initialize the E5-large model for embeddings.
    model = SentenceTransformer("intfloat/e5-large")
    
    # Process files and get new sentence results (skipping cached sentences)
    new_results = process_all_files(args.pdf_dir, model, args.tika_url, args.metadata_file)
    
    # Build (or update) the FAISS index incrementally
    build_incremental_index(new_results, model, args.index_file, args.metadata_file)
    
    # Interactive query loop using TF-IDF weighted aggregated search
    while True:
        query = input("\n🔎 Enter a search query (or 'exit' to quit): ").strip()
        if query.lower() == "exit":
            break
        aggregated_search_tfidf(query, model, args.index_file, args.metadata_file, top_k_sentences=50, top_k_resumes=5)
