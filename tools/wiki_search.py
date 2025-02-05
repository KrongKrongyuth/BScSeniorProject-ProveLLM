from langchain_community.docstore.document import Document
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun

def get_wiki_docs(query:str, top_k:int = 1):
    wiki_wrapper = WikipediaAPIWrapper(top_k_results=top_k, doc_content_chars_max=300)
    wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)
    
    docs = wiki_tool.invoke({'query': query})
    wiki_results = Document(page_content=docs)
    
    return wiki_results