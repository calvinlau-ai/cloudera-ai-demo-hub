import itertools
import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModel, AutoTokenizer
from milvus import default_server
from pymilvus import MilvusClient, DataType


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


def init_vector_database():
    uri = "http://localhost:%s" % default_server.listen_port
    client = MilvusClient(uri=uri)

    if 'rag4fin' not in client.list_collections():
        schema = MilvusClient.create_schema(
            auto_id=False,
            enable_dynamic_field=True,
        )

        schema.add_field(field_name="split_id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=768)

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="AUTOINDEX",
            metric_type="IP"
        )
        client.create_collection(
            collection_name="rag4fin",
            schema=schema,
            index_params=index_params
        )
    return client


class VectorIndex:
    def __init__(self, df, model):
        self.ids = None
        self.vecs = None
        self.df = df
        self.model = model
        self.reset()
        # self.milvus_client = init_vector_database()

    def reset(self):
        self.ids = []
        self.vecs = np.empty([0, 768], dtype=np.float32)

    def next_line(self):
        num_rows = self.df.shape[0]
        for i in range(num_rows):
            yield (i+1)*10000, self.df['title'][i]
            lines = [line for line in self.df['text'][i].split('\n') if len(line.strip()) > 0]
            for j, line in enumerate(lines, start=1):
                yield (i+1)*10000+j, line

    def splits(self, split_size):
        line_iter = self.next_line()
        while True:
            lines = list(itertools.islice(line_iter, split_size))
            if len(lines) == 0:
                break
            ids = [line[0] for line in lines]
            docs = [line[1] for line in lines]
            yield ids, docs

    def split_embedding(self, docs):
        vecs = self.model.encode(docs)
        self.vecs = np.concatenate((self.vecs, vecs))
        print('.', end='')

    def checkpoint(self, dir_prefix='index'):
        id_range = '%s-%s' % (self.ids[0], self.ids[-1])
        filename = '%s/split-ids-%s.npy' % (dir_prefix, id_range)
        np.save(filename, self.ids)
        filename = '%s/split-vecs-%s.npy' % (dir_prefix, id_range)
        np.save(filename, self.vecs)
        print(' Checkpoint %s' % self.ids[-1])

    def proc_docs(self, show_progress=200, ckpt_vecs=20000):
        for ids, docs in self.splits(show_progress):
            self.ids += ids
            self.split_embedding(docs)

            if len(self.ids) > 0 and len(self.ids) % ckpt_vecs == 0:
                self.checkpoint()
                self.reset()


# default_server.show_startup_banner = True
# default_server.start()

contriever_path = '/home/cdsw/models/contriever-msmarco'
model = MsmarcoModel(contriever_path)
df = pq.read_table('data/train-1-of-2.parquet').to_pandas()
vec_idx = VectorIndex(df, model)
vec_idx.proc_docs()
