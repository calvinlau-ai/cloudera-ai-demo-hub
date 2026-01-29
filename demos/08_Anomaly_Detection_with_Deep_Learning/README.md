# TKO Lab

## Fraud Detection

This is the CML port of the prototype which is part of the [Deep Learning for Anomaly Detection](https://ff12.fastforwardlabs.com/) report from Cloudera Fast Forward Labs.

## Auto Deploy

To build all the project artifacts, run the `0_cdsw-build.py` file in a Python3 session.

## Model Definition

This fraud detection model is defined as an AutoEncoder with [PyTorch](https://pytorch.org/). Here is the definition code:

```
class autoencoder(nn.Module):
    def __init__(self,num_input):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_input, 15),
            nn.ReLU(True),
            nn.Linear(15, 7))
        self.decoder = nn.Sequential(
            nn.Linear(7, 15),
            nn.ReLU(True),
            nn.Linear(15, num_input),
            nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
```

## Model Training

A trained model comes with the project, and can be directly used if you only need to present a demo. If you wish to retrain the model, use `1_fraud-model-train.ipynb`.

## Model Deploy

If you manually deploy the model into [Cloudera Machine Learning](https://www.cloudera.com/products/machine-learning.html), please use the `2_fraud-model-deploy.py` file and specify the `predict` function.

## Model Inference by CDSW

When inferencing, this deployed model accept an JSON-based HTTP/POST request. The main part of the request is composed of 3 fields:

* time:  the seconds elapsed between each transaction and the first transaction in the dataset.
* v: the 28-element list of the features obtained with PCA, i.e. V1, V2, … V28 of the dataset.
* amout: the transaction amout.

Here is a sample code snippet of requesting model inference in Python:

```
import requests

v = [-1.3598071336738,-0.0727811733098497,2.53634673796914,1.37815522427443,
    -0.338320769942518,0.462387777762292,0.239598554061257,0.0986979012610507,
    0.363786969611213,0.0907941719789316,-0.551599533260813,-0.617800855762348,
    -0.991389847235408,-0.311169353699879,1.46817697209427,-0.470400525259478,
    0.207971241929242,0.0257905801985591,0.403992960255733,0.251412098239705,
    -0.018306777944153,0.277837575558899,-0.110473910188767,0.0669280749146731,
    0.128539358273528,-0.189114843888824,0.133558376740387,-0.0210530534538215]
time = 0
amount = 149.62
request={"v": v, "time": time, "amount": amount}

data='{"accessKey":"mctpv79qnql7h9t1ncfeq7v57cwlu65k","request":request}'
endpoint = 'https://modelservice.ml-2a091c33-4ac.demo-aws.ylcu-atmi.cloudera.site/model'
r = requests.post(endpoint, data=data, headers={'Content-Type': 'application/json'})
```

## Model View

If you wish to get a quick view of the trained model, use `3_fraud-model-view.ipynb`.
