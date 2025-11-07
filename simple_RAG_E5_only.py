
import torch
from sentence_transformers import SentenceTransformer
from input_text import facts_me


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'


task = 'Given a query, retrieve relevant passages that answer the query'
queries = [
    get_detailed_instruct(task, 'Who am I?')
]

input_texts = queries + facts_me

model = SentenceTransformer('intfloat/multilingual-e5-large-instruct')

embeddings = model.encode(input_texts, convert_to_tensor=True, normalize_embeddings=True)
scores = (embeddings[0] @ embeddings[1:].T) * 100
print(scores.tolist())


# Print best match according to similarity of embeddings
best_idx = torch.argmax(scores).item()
best_fact = facts_me[best_idx]
best_score = scores[best_idx].item()

print(f"💬 Query: {queries}")
print(f"🤖 Answer: {best_fact}  (score: {best_score:.2f})")





