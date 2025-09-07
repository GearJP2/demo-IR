from sentence_transformers import SentenceTransformer, util


# Load statutes
with open("statutes.txt", "r", encoding="utf-8") as f:
    statutes = f.read().splitlines()

# User query (does not go in statutes file)
query = "tenant does does does not not not pay money to landlord"

# Load a pretrained model
model = SentenceTransformer('all-MiniLM-L6-v2')  # English version
# For Thai law, you could use "airesearch/wangchanberta-base-wiki-book" from HuggingFace

# Encode query + statutes
embeddings = model.encode(statutes, convert_to_tensor=True)
query_embedding = model.encode(query, convert_to_tensor=True)

# Compute cosine similarity
cosine_scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]

# Rank results
ranked = sorted(zip(statutes, cosine_scores), key=lambda x: x[1], reverse=True)
print("\nSentence-BERT Results:")
for doc, score in ranked:
    print(f"{score:.4f} | {doc}")
