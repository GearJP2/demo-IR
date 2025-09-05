from rank_bm25 import BM25Okapi
import nltk

# Load statutes
with open("statutes.txt", "r", encoding="utf-8") as f:
    statutes = f.read().splitlines()

# User query (does not go in statutes file)
query = "tenant does not pay rent"


# Tokenize text
nltk.download("punkt")
tokenized_statutes = [nltk.word_tokenize(doc.lower()) for doc in statutes]
bm25 = BM25Okapi(tokenized_statutes)

query_tokens = nltk.word_tokenize(query.lower())
scores = bm25.get_scores(query_tokens)

# Rank results
ranked = sorted(zip(statutes, scores), key=lambda x: x[1], reverse=True)
print("BM25 Results:")
for doc, score in ranked:
    print(f"{score:.2f} | {doc}")
