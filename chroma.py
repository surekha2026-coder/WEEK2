from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --------------------------------
# 1. Load Hugging Face Embeddings
# --------------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# --------------------------------
# 2. Sample documents
# --------------------------------
documents = [
    "Natural language processing is a field of artificial intelligence",
    "Machine learning allows computers to learn from data",
    "Chroma is a vector database used for embeddings",
    "Python is popular for AI and data science"
]

# --------------------------------
# 3. Create Chroma vector store
# --------------------------------
vectorstore = Chroma.from_texts(
    texts=documents,
    embedding=embedding_model,
    collection_name="hf_chroma_demo"
)

print("✅ Documents stored in Chroma")

# --------------------------------
# 4. Similarity search
# --------------------------------
query = "What is NLP?"

results = vectorstore.similarity_search(query, k=2)

print("\n🔍 Search Results:")
for doc in results:
    print("-", doc.page_content)
