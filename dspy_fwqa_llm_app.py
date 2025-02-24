import dspy
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS as LangchainFAISS
from langchain.embeddings import HuggingFaceEmbeddings

# 1. Custom Retriever for DSPy
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

# 2. Llama Language Model wrapper (updated for newer DSPy versions)
class LlamaLanguageModel(dspy.LanguageModel):
    def __init__(self, model_name="meta-llama/Llama-2-70b-chat-hf"):
        super().__init__()
        self.model_name = model_name
        self.max_tokens = 2048
        self.temperature = 0.7
        
        # Initialize model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
    def __call__(self, prompt, **kwargs):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=self.max_tokens,
            temperature=self.temperature,
            do_sample=True,
            **kwargs
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# 3. Setup DSPy with custom retriever
def setup_dspy(faiss_index_path, vector_db_path):
    llm = LlamaLanguageModel()
    retriever = CustomFAISSRetriever(faiss_index_path, vector_db_path)
    dspy.settings.configure(lm=llm, rm=retriever)

# 4. Enhanced RAG module
class MultilingualRAGChainOfThought(dspy.Module):
    def __init__(self):
        super().__init__()
        self.qa_chain = dspy.ChainOfThought(
            'context, question -> answer',
            prompt_template="""
            你是一個專業的AI助手，請根據提供的上下文來回答問題。請依照以下步驟思考：
            
            上下文資訊：{context}
            
            問題：{question}
            
            讓我們按步驟來：
            1) 先理解問題要點
            2) 從上下文中找出相關資訊
            3) 組織完整的回答
            
            回答："""
        )
        self.retrieve = dspy.Retrieve(k=3)
    
    def forward(self, question):
        retrieved_contexts = self.retrieve(question).passages
        context = "\n\n".join(retrieved_contexts)
        prediction = self.qa_chain(context=context, question=question)
        
        return {
            'answer': prediction.answer,
            'contexts': retrieved_contexts
        }

# 5. Main application setup
def create_rag_application(faiss_index_path, vector_db_path, model_name="meta-llama/Llama-2-70b-chat-hf"):
    setup_dspy(faiss_index_path, vector_db_path)
    rag = MultilingualRAGChainOfThought()
    return rag

# 6. Example usage
def main():
    try:
        rag = create_rag_application(
            faiss_index_path="qa_index.faiss",
            vector_db_path="tech_support_faiss"
        )
        
        # 測試問題
        question = "請問如何處理系統異常?"
        result = rag(question)
        
        print(f"問題: {question}")
        print(f"回答: {result['answer']}")
        print("\n參考來源:")
        for i, context in enumerate(result['contexts'], 1):
            print(f"{i}. {context[:200]}...")
            
    except Exception as e:
        print(f"錯誤: {str(e)}")

if __name__ == "__main__":
    main()
