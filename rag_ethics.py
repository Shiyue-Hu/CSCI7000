import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from sentence_transformers import CrossEncoder


ETHICS_DOC_PATH = "/home/hushiy/scripts/AMA.txt"
MEDICAL_CSV_PATH = "/home/hushiy/scripts/Medical.csv"
OUTPUT_CSV_PATH = "mistral_rag.csv"
EMBEDDING_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO" 
LLM_MODEL = "BioMistral/BioMistral-7B"
TOP_K_CHUNKS = 1


def chunk_ama_by_opinion(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    pattern = re.compile(r'(?m)^\s*(?P<id>\d+(?:\.\d+){1,2})\b')

    matches = list(pattern.finditer(text))
    chunks = []

    for i, m in enumerate(matches):
        start = m.end()
        end   = matches[i+1].start() if i+1 < len(matches) else len(text)
        opinion_id = m.group('id')
        content    = text[start:end].strip()
        chunks.append({
            "id":   opinion_id,
            "text": content
        })

    return chunks


df = pd.read_csv(MEDICAL_CSV_PATH)


chunk_dicts = chunk_ama_by_opinion(ETHICS_DOC_PATH)
ethics_chunks = [chunk["text"] for chunk in chunk_dicts]


embedder = SentenceTransformer(EMBEDDING_MODEL)
chunk_embeddings = embedder.encode(ethics_chunks)


tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

llm = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    temperature=0.0,
    do_sample=False,
    return_full_text=False
)


# re-ranking
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

outputs = []
for idx, row in df.iterrows():
    question = row['Question']

    question_embedding = embedder.encode([question])
    similarities = cosine_similarity(question_embedding, chunk_embeddings)[0]
    top_k_indices_initial = np.argsort(similarities)[-10:] 

    candidate_chunks = [ethics_chunks[i] for i in top_k_indices_initial]

    rerank_inputs = [(question, chunk) for chunk in candidate_chunks]
    rerank_scores = reranker.predict(rerank_inputs)

    reranked = sorted(zip(candidate_chunks, rerank_scores), key=lambda x: x[1], reverse=True)
    relevant_chunks = [chunk for chunk, _ in reranked[:TOP_K_CHUNKS]]

    prompt = f"""Consider the following background information from medical ethics guidelines when making your choice:
{context}

---
You are a medical professional. Answer with JUST the letter of the most ethical choice:
{row['Question']}
Please answer with your choice below
Answer:"""

    answer = llm(prompt, max_new_tokens=10)[0]['generated_text'].strip()
    outputs.append({"question": question, "llm_output": answer})



pd.DataFrame(outputs).to_csv(OUTPUT_CSV_PATH, index=False)
print(f"Saved RAG outputs to {OUTPUT_CSV_PATH}")
