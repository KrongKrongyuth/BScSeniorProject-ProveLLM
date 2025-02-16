import json
import pandas as pd

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.docstore.document import Document
from langchain_community.vectorstores import FAISS
from pathlib import Path
from dotenv import load_dotenv; load_dotenv()

def get_eval_template(raw_data_path:str, only_def:bool = True, save_loc:str = ''):
    eval_df = pd.DataFrame(columns=[
        'theorem_id',
        'throrem_titles',
        'theorem_contents',
        'theorem_refs',         # Actual 'reference title' from grounds truth
        'ref_ids',              # Actual 'references ID' from grounds truth
        'retrieved_3',          # Top 3 retrieved documents ID and L2 score from DB
        'retrieved_5',          # Top 5 retrieved documents ID and L2 score from DB
        'retrieved_10',         # Top 10 retrieved documents ID and L2 score from DB
        'retrieved_50',         # Top 50 retrieved documents ID and L2 score from DB
        'P@3',                  # Precision @ 3
        'P@5',                  # Precision @ 5
        'P@10',                 # Precision @ 10
        'P@50',                 # Precision @ 50
        'R@3',                  # Recall @ 3
        'R@5',                  # Recall @ 5
        'R@10',                 # Recall @ 10
        'R@50',                 # Recall @ 50
        'self_retrieve'         # Check whether self-retrieve 
        ])
    raw_data = json.loads(Path(raw_data_path).read_text())
    
    for _, category in enumerate(raw_data['dataset']):
        if category == 'theorems':
            theorem_id, title, contents, refs, ref_ids = [], [], [], [], []
            for _, theorem in enumerate(raw_data['dataset'][category]):
                theorem_ref = len(list(filter(lambda x: x<19734, theorem['ref_ids'])))
                
                if only_def:
                    if theorem['ref_ids'] and theorem['contents'] and theorem_ref == 0:
                        theorem_id.append(theorem['id'])
                        title.append(theorem['title'])
                        contents.append(''.join(theorem['contents']))
                        refs.append(theorem['refs'])
                        ref_ids.append(theorem['ref_ids'])
                elif not only_def:
                    if theorem['ref_ids'] and theorem['contents']:
                        theorem_id.append(theorem['id'])
                        title.append(theorem['title'])
                        contents.append(''.join(theorem['contents']))
                        refs.append(theorem['refs'])
                        ref_ids.append(theorem['ref_ids'])
            
            eval_df = pd.concat([eval_df,
                                    pd.DataFrame({
                                        'theorem_id': theorem_id,
                                        'throrem_titles': title,
                                        'theorem_contents': contents,
                                        'theorem_refs': refs,
                                        'ref_ids': ref_ids})],
                                    ignore_index=True)
    
    if save_loc: eval_df.to_csv(f'{save_loc}')
    
    return eval_df

def get_corpus(raw_data_path:str, only_def:bool = True):
    corpus = []
    raw_data = json.loads(Path(raw_data_path).read_text())
    
    for _, category in enumerate(raw_data['dataset']):
        if category != 'retrieval_examples' and not only_def:
            for _, data in enumerate(raw_data['dataset'][category]):
                id = data['id']
                title = data['title']
                content = ''.join(data['contents'])
                
                document = f"""{title}: {content}"""
                corpus.append(Document(
                    page_content = document,
                    metadata = {'source': category, 'id': id}
                    ))
        if category not in ['theorems', 'retrieval_examples'] and only_def:
            for _, data in enumerate(raw_data['dataset'][category]):
                id = data['id']
                title = data['title']
                content = ''.join(data['contents'])
                
                document = f"""{title}: {content}"""
                corpus.append(Document(
                    page_content = document,
                    metadata = {'source': category, 'id': id}
                    ))
    
    return corpus

def save_corpus_to_index(corpus:list[Document], embedding_model:HuggingFaceEmbeddings, save_loc:str):
    refs_db = FAISS.from_documents(corpus, embedding_model)
    refs_db.save_local(folder_path=save_loc,
                    index_name='ref_index')

def get_index(index_loc:str, embedding_model:HuggingFaceEmbeddings):
    refs_db = FAISS.load_local(
        folder_path=index_loc,
        embeddings=embedding_model,
        index_name='ref_index',
        allow_dangerous_deserialization=True)

    return refs_db

def retrieval_eval(eval_df:pd.DataFrame, index:FAISS, save_loc:str = ''):
    for _, data in enumerate(eval_df.iterrows()):
        theorem_query = data[1]['theorem_contents']
        actual_refs = set(data[1]['ref_ids'])
        
        retrieved_docs = index.similarity_search_with_score(
            query = theorem_query,
            k = 50
            )
        
        if eval_df.loc[data[0], 'theorem_id'] in [doc[0].metadata['id'] for doc in retrieved_docs]:
            self_index = [doc[0].metadata['id'] for doc in retrieved_docs].index(eval_df.loc[data[0], 'theorem_id'])
            retrieved_docs = index.similarity_search_with_score(
                query = theorem_query,
                k = 51
                )
            retrieved_docs.pop(self_index)
        
        for k in [3, 5, 10, 50]:
            result_dict = {doc[0].metadata['id']:doc[1] for doc in retrieved_docs[:k]}
            
            if eval_df.loc[data[0], 'theorem_id'] in result_dict.keys():
                eval_df.at[data[0], 'self_retrieve'] = True
            else:
                eval_df.at[data[0], 'self_retrieve'] = False
            
            retrieve_ele = set(result_dict.keys())
            intersect_k = actual_refs.intersection(retrieve_ele)
            
            precision_K = len(intersect_k)/k
            recall_K = len(intersect_k)/len(actual_refs)
            
            eval_df.at[data[0], f'retrieved_{k}'] = result_dict
            eval_df.at[data[0], f'P@{k}'] = precision_K
            eval_df.at[data[0], f'R@{k}'] = recall_K
            
    if save_loc: eval_df.to_csv(f'{save_loc}')
        
    return eval_df

if __name__ == "__main__":
    print(get_eval_template(
        raw_data_path='/Users/ton_kkrongyuth/Senior Project/Aj.Pannapa/Prove_LLM/Data/NATURALPROOFS_DATASET/naturalproofs_proofwiki.json',
        only_def=True)
        )