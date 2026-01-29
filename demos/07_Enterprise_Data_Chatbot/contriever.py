import os
import pyarrow.parquet as pq
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from utils import log


class Retriever:
    def __init__(self):
        model_path = os.getenv("EMB_MODEL")
        log.info('Loading embedding model %s ... ' % model_path)
        self.model = SentenceTransformer(model_path)
        pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        self.index = pc.Index(os.getenv('PINECONE_INDEX_NAME'))
        # self.texts = self.load_texts()
        self.df = pq.read_table(os.getenv('DATA_FILE')).to_pandas()

    def text_id(self, user_input, top_k=3):
        '''
        Returns a list of 2-tuples: [(id1, score1), (id2, score2), (id3, score3), ...]
        1st element of the tuple: ID.
        2nd element of the tuple: Matching score.
        '''
        query = self.model.encode([user_input])[0].tolist()
        results = self.index.query(vector=query, top_k=top_k, include_values=True)['matches']
        return [(int(r['id'].split('-')[1]), r['score']) for r in results]

    def get_texts(self, user_input, top_k=3):
        text_ids = self.text_id(user_input, top_k)
        titles, texts = self.df['title'], self.df['text']
        return [(id, titles[id], texts[id], score) for id, score in text_ids]
