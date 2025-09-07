from sentence_transformers import SentenceTransformer, util
import openai  # if you want GPT generation (replace with your model call)

# 1. Statutes database (can later load from CSV)
statutes = [
    "Section 563: The lessee is bound to pay the rent at the agreed time.",
    "Section 420: A person who unlawfully injures the rights of another is bound to make compensation.",
    "Section 151: A contract is void if it is made under mistake, fraud or duress.",
    "Section 653: The borrower is bound to repay the loan at the due time."
]

# 2. Encode with BERT (use Legal-BERT later)
model = SentenceTransformer("all-MiniLM-L6-v2")  
embeddings = model.encode(statutes, convert_to_tensor=True)

# User query
query = "tenant did not pay the landlord"
query_emb = model.encode(query, convert_to_tensor=True)

# 3. Find most relevant statutes
cosine_scores = util.pytorch_cos_sim(query_emb, embeddings)[0]
top_result_idx = int(cosine_scores.argmax())
retrieved_statute = statutes[top_result_idx]

print("Most relevant statute:")
print(retrieved_statute)

# 4. Generation step (pseudo-code)
# Example using GPT — replace with your model of choice
"""
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a legal assistant."},
        {"role": "user", "content": f"Generate a legal case scenario based on: {retrieved_statute}"}
    ]
)
print(response["choices"][0]["message"]["content"])
"""
