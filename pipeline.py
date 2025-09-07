import os
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

# 1. Load statutes from CSV
df = pd.read_csv("statutes.csv")
statutes = df["text"].tolist()

# 2. Load embedding model (start with general, swap to Legal-BERT later)
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(statutes, convert_to_tensor=True)

# 3. User query
query = "tenant refused to pay landlord for 3 months"
query_emb = model.encode(query, convert_to_tensor=True)

# 4. Rank results by cosine similarity
cosine_scores = util.pytorch_cos_sim(query_emb, embeddings)[0]

# Get top-k (e.g., top 2)
k = 2
top_results = cosine_scores.topk(k)

print("🔎 Top relevant statutes:")
for score, idx in zip(top_results[0], top_results[1]):
    law_id = df.iloc[int(idx)]["id"]
    law_text = df.iloc[int(idx)]["text"]
    print(f"ID {law_id} | Score: {score:.4f} | {law_text}")

# 5. OPTIONAL: Send top statutes to GPT for scenario generation

retrieved_text = " ".join([df.iloc[int(i)]['text'] for i in top_results[1]])

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",   # or "gpt-4-0613" if you have GPT-4 access
    messages=[
        {"role": "system", "content": "You are a legal assistant that generates example scenarios for legal education."},
        {"role": "user", "content": f"Based on these statutes: {retrieved_text}\nGenerate a short legal training case scenario."}
    ]
)


print("\n📘 Generated Scenario:")
print(response["choices"][0]["message"]["content"])

