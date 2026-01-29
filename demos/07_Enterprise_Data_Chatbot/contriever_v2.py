import os
import pyarrow.parquet as pq
import torch
from transformers import AutoModel, AutoTokenizer
from FlagEmbedding import FlagReranker

from milvus import default_server
from pymilvus import MilvusClient, DataType
from utils import log

default_server.show_startup_banner = True
default_server.start()


class MsmarcoModel:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to('cuda')
    
    def mean_pooling(self, token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings
    
    def encode(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to('cuda')
        with torch.no_grad():
            outputs = self.model(**inputs)
            return self.mean_pooling(outputs[0], inputs['attention_mask']).detach().cpu().numpy()


class Retriever:
    def __init__(self):
        model_path = os.getenv("EMB_MODEL")
        log.info('Loading embedding model %s ... ' % model_path)
        self.model = MsmarcoModel(model_path)
        self.reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
        self.milvus_client = MilvusClient(uri=os.getenv("MILVUS_URI"))
        self.df = pq.read_table(os.getenv('DATA_FILE')).to_pandas()

    def combine_candidates(self, candidates):
        ids_out = []
        for k in candidates[0]:
            # if len(ids_out) > 10:
            #     break
            id = k['id']
            distance = k['distance']
            if distance < 0.5:
                break
            elif id % 10000 == 0:  # The title is matched
                doc_id = id//10000-1
                ids_out.append(doc_id)
            else:
                ids_out.append((id//10000-1, id%10000-1))
        for i in range(len(ids_out)):
            id = ids_out[i]
            if isinstance(id, int):
                for j in range(len(ids_out)):
                    if ids_out[j] is not None and not isinstance(ids_out[j], int) and ids_out[j][0] == id:
                        ids_out[j] = None
        return [id for id in ids_out if id is not None]
                
    def text_id(self, user_input, top_k=100):
        '''
        Returns a list of 2-tuples: [(id1, score1), (id2, score2), (id3, score3), ...]
        1st element of the tuple: ID.
        2nd element of the tuple: Matching score.
        '''
        query = self.model.encode([user_input])
        candidates = self.milvus_client.search(
            collection_name=os.getenv("MILVUS_INDEX_NAME"),
            data=query,
            limit=top_k, # Max. number of search results to return
            search_params={"metric_type": "COSINE", "params": {}} # Search parameters
        )
        candidates = self.combine_candidates(candidates)
        return candidates

    def rerank(self, user_input, candidates, top_k=3):
        print('Candidate 1 #:', len(candidates))
        candidates = [id for id in candidates
                      if isinstance(id, int)
                      or len(self.df['text'][id[0]].split('\n')[id[1]].split())>3]
        print('Candidate 2 #:', len(candidates))
        # docs = [df['title'][id] + '\n' + df['text'][id] 
        #         if isinstance(id, int) else
        #         df['text'][id[0]].split('\n')[id[1]]
        #         for id in candidates]
        candidates = list(set([id if isinstance(id, int) else id[0] for id in candidates]))
        print('Candidate 3 #:', len(candidates))
        docs = [self.df['title'][id] + '\n' + self.df['text'][id] for id in candidates]
        query_and_docs = [[user_input, doc] for doc in docs]
        scores = self.reranker.compute_score(query_and_docs, normalize=True)
        return [x for x in zip(candidates, docs, scores) if x[2]>0.4]

    def get_texts(self, user_input, top_k=3):
        # text_ids = self.text_id(user_input, top_k)
        # titles, texts = self.df['title'], self.df['text']
        # return [(id, titles[id], texts[id], score) for id, score in text_ids]
        candidates = self.text_id(user_input, top_k=top_k*80)
        return self.rerank(user_input, candidates, top_k)
    
    def get_texts_without_rerank(self, user_input, top_k=20):
        candidates = self.text_id(user_input, top_k=top_k)
        docs = [df['title'][id] + '\n' + df['text'][id] 
                if isinstance(id, int) else
                df['text'][id[0]].split('\n')[id[1]]
                for id in candidates]
        return list(zip(candidates, docs))


#from dotenv import load_dotenv
#load_dotenv(dotenv_path='mydata-chatbot.env')
#r=Retriever()
#r.get_texts('Who wins prime contract of the U.S. Navy?')
#r.get_texts('What is the revenue of Microsoft?')
#r.get_texts('What is the revenue of Microsoft in game?')
#r.get_texts('What is the revenue of Cloudera?')
