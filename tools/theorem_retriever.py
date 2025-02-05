from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

def get_retriever():
    """
    Load and create the retriever
    """
    db_folder_path = '/Users/ton_kkrongyuth/Senior Project/Aj.Pannapa/Prove_LLM/Data/theorem_100_db'
    
    embedding_llm = OllamaEmbeddings(model='llama3.1:8b')
    theorem_db = FAISS.load_local(
        folder_path = db_folder_path,
        index_name = 'theorem_100_db',
        allow_dangerous_deserialization = True,
        embeddings = embedding_llm
    )
    theorem_retriever = theorem_db.as_retriever()
    
    return theorem_retriever

def retrieve_docs(query:str):
    """
    Return the retrieved documents
    """
    retriever = get_retriever()
    
    return retriever.invoke(query)