import os
import json
import requests
import spacy
import re
from tqdm import tqdm

def extract_clean_text(file_path, tika_url="http://localhost:9998/rmeta/text"):
    """
    Extract main textual content using Tika's /rmeta/text endpoint.
    """
    try:
        with open(file_path, "rb") as f:
            headers = {"Accept": "application/json"}
            response = requests.put(tika_url, data=f, headers=headers)
            response.raise_for_status()
            content_json = response.json()
            if content_json and isinstance(content_json, list):
                main_text = content_json[0].get("X-TIKA:content", "")
                return main_text.strip()
            else:
                print(f"⚠️ Unexpected response for {file_path}")
    except Exception as e:
        print(f"❌ Error extracting from {file_path}: {e}")
    return ""

def clean_sentence(text):
    """
    Clean sentence by removing bullets, line breaks, and excessive whitespace.
    """
    text = re.sub(r'[\n\r]+', ' ', text)         # Replace newlines with space
    text = re.sub(r'[•]', '', text)             # Remove bullet characters
    text = re.sub(r'\s{2,}', ' ', text)          # Collapse multiple spaces
    return text.strip()

def split_into_sentences(text, nlp):
    """
    Split text into clean, meaningful sentences using spaCy.
    """
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip() and len(sent.text.strip()) > 5]

def process_resumes(pdf_dir, output_json, tika_url="http://localhost:9998/rmeta/text"):
    """
    Process all PDF resumes, extract main content and split into cleaned sentences.
    """
    results = []
    supported_exts = ('.pdf', '.doc', '.docx', '.pptx', '.txt')
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(supported_exts)]
    
    if not pdf_files:
        print(f"📂 No PDF files found in {pdf_dir}")
        return

    nlp = spacy.load("en_core_web_sm")

    for pdf_file in tqdm(pdf_files, desc="🔍 Processing PDFs"):
        file_path = os.path.join(pdf_dir, pdf_file)
        text = extract_clean_text(file_path, tika_url)
        if not text:
            continue
        sentences = split_into_sentences(text, nlp)
        for sentence in sentences:
            cleaned = clean_sentence(sentence)
            if cleaned:
                results.append({
                    "file": pdf_file,
                    "sentence": cleaned,
                    "length": len(cleaned)
                })

    try:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Processed {len(results)} sentences. Output saved to {output_json}")
    except Exception as e:
        print(f"❌ Error saving output JSON: {e}")

if __name__ == "__main__":
    pdf_directory = "resumes"                      # Folder containing PDFs
    output_file = "resumes_main_content.json"      # Output JSON
    tika_server_url = "http://localhost:9998/rmeta/text"

    process_resumes(pdf_directory, output_file, tika_url=tika_server_url)
