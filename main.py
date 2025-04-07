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
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

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
# Initialize TinyLlama for text generation (justification)
# Use the full model repo ID. Note that this model may still be heavy for CPU-only setups.
tinyllama_tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
tinyllama_model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.float32).eval()

def generate_justification(query, sentence):
    try:


        """
        Generate a justification explaining why the resume sentence matches the query.
        """
        prompt = (
            f"You are an AI assistant helping recruiters.\n"
            f"Job Query: {query}\n"
            f"Resume Sentence: {sentence}\n"
            "Explain in 1-2 concise sentences why this sentence is relevant."
        )
        #inputs = tinyllama_tokenizer(prompt, return_tensors="pt")
        #inputs = {k: v.to('cpu') for k, v in inputs.items()}
        outputs = tinyllama_model.generate(**inputs, max_new_tokens=100, do_sample=True, top_k=50)
        justification = tinyllama_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return justification.strip()
    except Exception as e:
        print(f"❌ Error generating justification: {e}")        
        return "Justification generation failed."
    except RuntimeError as e:                   
        if "out of memory" in str(e):           
            print("❌ Model is too large for this setup. Please use a smaller model or GPU.")
            return "Justification generation failed due to memory constraints."

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
    # Use ThreadPoolExecutor to ensure the model is loaded only once.
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
def search(query, model, index_file, metadata_file, top_k=5):
    query_vec = model.encode(f"query: {query}", normalize_embeddings=True).astype("float32")
    index = faiss.read_index(index_file)
    
    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    distances, indices = index.search(np.array([query_vec]), top_k)
    print(f"\n🔍 Top {top_k} results for query: '{query}'\n")
    for i, idx in enumerate(indices[0]):
        result = metadata[idx]
        justification = generate_justification(query, result["sentence"])
        print(f"{i+1}.  (File: {result['file']}), Distance: {distances[0][i]:.3f}")
        print(f"   💬 Justification: {justification}\n")

# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Incremental FAISS-based Resume Search Engine")
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
    
    # Interactive query loop
    while True:
        query = input("\n🔎 Enter a search query (or 'exit' to quit): ").strip()
        if query.lower() == "exit":
            break
        search(query, model, args.index_file, args.metadata_file, top_k=5)
