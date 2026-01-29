import fasttext

model = fasttext.load_model('lid.176.ftz')
pred = model.predict('Berapa pendapatan Cloudera pada tahun 2019?', k=1)[0][0]
print(pred)
