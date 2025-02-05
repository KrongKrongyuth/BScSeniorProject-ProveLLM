from typing import List

from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv; load_dotenv()

def get_response(query:str, context:List[str]):
    system_prompt = """
    You are "Santi" a helpful AI for helping mathematicians prove mathematical statements, theorems, questions, etc. 
    You must provide the proof straight to the point with the least confusion in your prove. 
    If you are not sure how to prove the given question, you should tell the human about what makes you not sure about how to prove it.
    
    You can use the following context as a refference to constrcut the correct proof.
    <context>
    {context}
    </context>
    
    If you have to answer in the mathematical notation you must answer in the LaTex format.
    """
    human_prompt = """
    Question: {question}
    """
    
    template = ChatPromptTemplate.from_messages(
        [
            ('system', system_prompt),
            ('human', human_prompt)
        ]
    )
    llm = ChatOllama(model='llama3.1:8b')
    output_parsers =  StrOutputParser()
    chain = template|llm|output_parsers
    
    return chain.invoke({'question': query, 'context': context})

if __name__ == "__main__":
    print(get_response("Can you give me a prove of Close Set"))