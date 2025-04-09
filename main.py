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
from collections import defaultdict

# -----------------------------
# Resume Processing and Indexing Components
# -----------------------------

# Lazy load spaCy model
nlp_model = None
def get_nlp():
    global nlp_model
    if nlp_model is None:
        nlp_model = spacy.load("en_core_web_sm")
    return nlp_model

# Compute MD5 hash for a file (to detect changes)
def compute_file_hash(file_path):
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
    except Exception as e:
        print(f"Error computing hash for {file_path}: {e}")
    return hash_md5.hexdigest()

# Compute MD5 hash for a sentence to avoid re-embedding duplicates
def compute_sentence_hash(sentence):
    return hashlib.md5(sentence.encode("utf-8")).hexdigest()

# Extract text from document using Tika (make sure Tika server is running)
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

# Clean up individual sentences
def clean_sentence(sentence):
    return sentence.replace('\n\n', ' ').replace('\n', ' ').replace('', '').strip()

# Split text into sentences using spaCy
def split_into_sentences(text, nlp):
    doc = nlp(text)
    return [clean_sentence(sent.text).strip() for sent in doc.sents if sent.text.strip() and len(sent.text.strip()) > 5]

# Process a single file: extract text, split into sentences, and compute embeddings
def process_file(file, pdf_dir, model, tika_url, existing_sentence_hashes):
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
            "sentence_hash": sent_hash
        })
    if new_sentences:
        passages = [f"passage: {s}" for s in new_sentences]
        embeddings = model.encode(passages, normalize_embeddings=True)
        for idx, emb in enumerate(embeddings):
            results[idx]["embedding"] = emb.tolist()
    else:
        results = []
    return results

# Load existing metadata from JSON (if available)
def load_existing_metadata(metadata_file):
    if os.path.exists(metadata_file):
        with open(metadata_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

# Process all files concurrently
def process_all_files(pdf_dir, model, tika_url, metadata_file):
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

# Build or update the FAISS index with new sentence embeddings
def build_incremental_index(new_results, model, index_file, metadata_file):
    existing_metadata = load_existing_metadata(metadata_file)
    all_metadata = existing_metadata.copy()
    sample_vec = model.encode("example", normalize_embeddings=True)
    dim = sample_vec.shape[0]
    if os.path.exists(index_file):
        index = faiss.read_index(index_file)
    else:
        index = faiss.IndexFlatIP(dim)  # cosine similarity using normalized vectors
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

# Update metadata with TF-IDF scores for lexical importance
def update_metadata_with_tfidf(metadata):
    sentences = [entry["sentence"] for entry in metadata]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences)
    tfidf_weights = tfidf_matrix.sum(axis=1).A1
    for entry, weight in zip(metadata, tfidf_weights):
        entry["tfidf_weight"] = weight
    return metadata

# -----------------------------
# RAG-Augmented Search Components
# -----------------------------

def build_contextual_chunks(metadata, window_size=2):
    """Create contextual chunks with surrounding sentences for better context"""
    chunks = []
    for i, item in enumerate(metadata):
        start = max(0, i - window_size)
        end = min(len(metadata), i + window_size + 1)
        context = " ".join([metadata[j]["sentence"] for j in range(start, end)])
        chunks.append({
            "text": context,
            "source": item["file"],
            "sentence_hash": item["sentence_hash"]
        })
    return chunks

class RAGRetriever:
    def __init__(self, index_file, metadata_file, model):
        self.index = faiss.read_index(index_file)
        with open(metadata_file, "r", encoding="utf-8") as f:          
          self.metadata = json.load(f)

        # Build contextual chunks from raw metadata
        self.chunks = build_contextual_chunks(self.metadata)
        self.model = model
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self._fit_tfidf()
    
    def _fit_tfidf(self):
        """Precompute TF-IDF matrix during initialization"""
        self.tfidf_matrix = self.vectorizer.fit_transform([c["text"] for c in self.chunks])
    
    def hybrid_score(self, query_vec, text):
        """Calculate hybrid score combining semantic and lexical relevance"""
        # Semantic score from FAISS (using cosine similarity)
        semantic_score = float(np.dot(query_vec, self.model.encode(text)))

        # Lexical score from TF-IDF (higher sum = more importance)
        lexical_score = self.vectorizer.transform([text]).sum(axis=1).A1[0]
        return 0.7 * semantic_score + 0.3 * lexical_score
    
    def retrieve(self, query, top_k=5):
        """Retrieve contextually relevant chunks with hybrid scoring"""
        query_vec = self.model.encode(f"query: {query}", normalize_embeddings=True).reshape(1, -1)
        # First-stage FAISS retrieval; we retrieve more candidates than needed.
        distances, indices = self.index.search(query_vec, top_k*3)
        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            hybrid = self.hybrid_score(query_vec, chunk["text"])
            results.append((hybrid, chunk))
        # Sort by hybrid score and deduplicate by sentence_hash.
        seen = set()
        final_results = []
        for hybrid, chunk in sorted(results, key=lambda x: x[0], reverse=True):
            if chunk["sentence_hash"] not in seen:
                seen.add(chunk["sentence_hash"])
                final_results.append(chunk)
            if len(final_results) >= top_k:
                break
        return final_results

# -----------------------------
# Enhanced RAG Generation with Source Validation
# -----------------------------
def rag_generation(query, context_chunks, api_key):
    """Generate answer with strict source validation and citation"""
    context_str = "\n\n".join([
        f"Excerpt {i+1} from {chunk['source']}:\n{chunk['text']}" 
        for i, chunk in enumerate(context_chunks)
    ])
    prompt = f"""**Job Requirement:** {query}

**Relevant Resume Excerpts:**
{context_str}

**Task:** 
1. Analyze if the candidate meets the requirement based ONLY on the excerpts.
2. Identify SPECIFIC matching qualifications with source citations (e.g., [Source 1], [Source 2]).
3. If no match exists, clearly state this.

**Response Guidelines:**
- Cite with [Source #] for each claim.
- Do not invent any qualifications.
- Highlight technical competencies and compare relevant experience."""
    
    try:
        response = google_text_generation(prompt, api_key, temperature=0.1)
        return response
    except Exception as e:
        return f"Generation error: {str(e)}"

# -----------------------------
# RAG-enhanced end-to-end search function
# -----------------------------
def rag_enhanced_search(query, model, index_file, metadata_file, top_k=5, api_key=""):
    """End-to-end RAG-powered search with verification"""
    retriever = RAGRetriever(index_file, metadata_file, model)
    context_chunks = retriever.retrieve(query, top_k=top_k)
    if not context_chunks:
        return "No relevant qualifications found in resumes."
    
    # Group retrieved chunks by resume file
    resume_contexts = defaultdict(list)
    for chunk in context_chunks:
        resume_contexts[chunk['source']].append(chunk)
    
    results = []
    for resume, chunks in resume_contexts.items():
        justification = rag_generation(query, chunks, api_key)
        results.append({
            "resume": resume,
            "match_score": len(chunks),
            "justification": justification,
            "source_chunks": chunks
        })
    
    # Sort results by number of matching chunks (as a proxy for match strength)
    return sorted(results, key=lambda x: x["match_score"], reverse=True)

# -----------------------------
# Google LLM Generation Function (using Gemini-2.0-Flash)
# -----------------------------
def google_text_generation(prompt, api_key, 
                           endpoint="https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
                           temperature=0.2, max_output_tokens=256):
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
            "topP": 0.95
        }
    }
    headers = {"Content-Type": "application/json"}
    params = {"key": api_key}
    try:
        response = requests.post(endpoint, headers=headers, params=params, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()
        return result["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        print(f"❌ Error during Google LLM text generation: {e}")
        return ""

# -----------------------------
# Main Function
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RAG-Enhanced Resume Search Engine: Retrieval Augmented Generation with resume-based justifications."
    )
    parser.add_argument("--pdf_dir", type=str, default="resumes", help="Directory containing resume files")
    parser.add_argument("--tika_url", type=str, default="http://localhost:9998/rmeta/text", help="Tika endpoint URL")
    parser.add_argument("--index_file", type=str, default="faiss.index", help="FAISS index file")
    parser.add_argument("--metadata_file", type=str, default="faiss_metadata.json", help="Metadata mapping file")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the FAISS index from scratch")
    parser.add_argument("--google_api_key", type=str, default="", help="Google API key for text generation")
    args = parser.parse_args()
    
    # For testing, you may temporarily hardcode the API key (remove for production)
    # args.google_api_key = "YOUR_GOOGLE_API_KEY"
    
    # If rebuilding, remove existing index and metadata.
    if args.rebuild:
        if os.path.exists(args.index_file):
            os.remove(args.index_file)
        if os.path.exists(args.metadata_file):
            os.remove(args.metadata_file)
        print("🔄 Rebuild selected: existing index and metadata removed.")
    
    # Initialize the SentenceTransformer model.
    model = SentenceTransformer("intfloat/e5-large")
    
    # Process resume files and update index/metadata.
    new_results = process_all_files(args.pdf_dir, model, args.tika_url, args.metadata_file)
    build_incremental_index(new_results, model, args.index_file, args.metadata_file)
    
    # Interactive RAG search loop.
    print("\nEnter a job requirement or position to search resumes with RAG-enhanced justifications.")
    print("Prefix with 'rag:' to use the RAG mode directly. (Or type 'exit' to quit)")
    while True:
        query = input("\n🔎 Enter a query (or 'exit'): ").strip()
        if query.lower() == "exit":
            break
        # If query begins with "rag:", use the RAG-powered generation.
        if query.lower().startswith("rag:"):
            actual_query = query[4:].strip()
            if not args.google_api_key:
                print("❌ Google API key not provided. Use the --google_api_key argument.")
                continue
            response = rag_enhanced_search(actual_query, model, args.index_file, args.metadata_file, top_k=5, api_key=args.google_api_key)
            print(f"\n🤖 RAG-Enhanced Search Results for: {actual_query}")
            for idx, result in enumerate(response, 1):
                print(f"\n🏆 Match #{idx}: {result['resume']}")
                print(f"📈 Match Strength (excerpts): {result['match_score']}")
                print(f"📝 Justification:\n{result['justification']}")
                print("🔗 Supporting Context:")
                for chunk in result["source_chunks"]:
                    print(f"   - From {chunk['source']}: {chunk['text'][:150]}...")
        else:
            # If not in explicit RAG mode, you may want to run your fallback aggregated search
            # (previously implemented method with per-resume justifications)
            # For simplicity, here we'll use the RAG function.
            if not args.google_api_key:
                print("❌ Google API key not provided. Use the --google_api_key argument.")
                continue
            response = rag_enhanced_search(query, model, args.index_file, args.metadata_file, top_k=5, api_key=args.google_api_key)
            print(f"\n🤖 RAG-Enhanced Search Results for: {query}")
            for idx, result in enumerate(response, 1):
                print(f"\n🏆 Match #{idx}: {result['resume']}")
                print(f"📈 Match Strength (excerpts): {result['match_score']}")
                print(f"📝 Justification:\n{result['justification']}")
                print("🔗 Supporting Context:")
                for chunk in result["source_chunks"]:
                    print(f"   - From {chunk['source']}: {chunk['text'][:150]}...")
