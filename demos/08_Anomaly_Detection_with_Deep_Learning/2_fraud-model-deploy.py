from datetime import datetime
import sys
sys.path.append('/home/cdsw/.local/lib/python3.6')
print(sys.path)

import torch
import torch.nn as nn

class autoencoder(nn.Module):
    def __init__(self,num_features):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_features, 15),
            nn.ReLU(True),
            nn.Linear(15, 7))
        self.decoder = nn.Sequential(
            nn.Linear(7, 15),
            nn.ReLU(True),
            nn.Linear(15, num_features),
            nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

num_features=30
split_point=-1.5

import pickle
file=open('min-max-scaler.pkl','rb')
scaler=pickle.load(file)
file.close()

model = autoencoder(num_features)
model.load_state_dict(torch.load('creditcard-fraud-minmaxscale-2.model'))
model.eval()

def predict(args:dict):
    with torch.no_grad():
        inp=[args['time']]+args['v']+[args['amount']]
        inp=scaler.transform([inp])
        inp=torch.tensor(inp, dtype=torch.float32)
        outp=model(inp)
        loss=torch.sum((inp-outp)**2,dim=1).sqrt().log()
        return loss.item()<split_point

