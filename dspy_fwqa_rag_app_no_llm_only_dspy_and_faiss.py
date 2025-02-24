import dspy
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS as LangchainFAISS
from langchain.embeddings import HuggingFaceEmbeddings

# 1. Custom Retriever for DSPy that uses your existing FAISS setup
class CustomFAISSRetriever(dspy.Retrieve):
    def __init__(self, faiss_index_path, vector_db_path, k=3):
        super().__init__()
        self.k = k
        # Load the FAISS index
        self.index = faiss.read_index(faiss_index_path)
        # Load the vector store
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )
        self.vector_db = LangchainFAISS.load_local(
            vector_db_path,
            self.embeddings
        )
        # Initialize the sentence transformer
        self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    
    def __call__(self, query):
        # Encode the query
        query_embedding = self.model.encode(
            query,
            convert_to_tensor=False,
            show_progress_bar=False
        )
        
        # Search in vector DB
        docs = self.vector_db.similarity_search_with_score(query, k=self.k)
        
        # Format results
        passages = []
        for doc, score in docs:
            context = doc.page_content
            metadata = doc.metadata
            formatted_context = f"{context}\nSource: {metadata['source']}\nLast Updated: {metadata['last_updated']}"
            passages.append(formatted_context)
        
        return dspy.Prediction(passages=passages)

# 2. Setup DSPy with custom retriever
def setup_retriever(faiss_index_path, vector_db_path):
    retriever = CustomFAISSRetriever(faiss_index_path, vector_db_path)
    return retriever

# 3. Simple search function
def search_similar_questions(retriever, question):
    results = retriever(question)
    return results.passages

def main():
    try:
        # Initialize retriever
        retriever = setup_retriever(
            faiss_index_path="qa_index.faiss",
            vector_db_path="tech_support_faiss"
        )
        
        # Test question
        question = "請問如何處理系統異常?"
        results = search_similar_questions(retriever, question)
        
        print(f"問題: {question}")
        print("\n相關文件:")
        for i, result in enumerate(results, 1):
            print(f"\n--- 文件 {i} ---")
            print(result)
            
    except Exception as e:
        print(f"錯誤: {str(e)}")

if __name__ == "__main__":
    main()
