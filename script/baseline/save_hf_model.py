from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
model.save("../../semeval-2026-task-4-baselines/checkpoints/all-minilm-l6-v2")