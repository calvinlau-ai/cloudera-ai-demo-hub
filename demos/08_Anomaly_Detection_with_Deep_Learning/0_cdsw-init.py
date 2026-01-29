## Run this file to auto deploy the model.

### Install the requirements
!bash cdsw-build.sh

### Download the data file and save it in the specified directory
!unzip -o data/creditcardfraud.zip

from utils.cmlapi import CMLApi
from datetime import datetime
import os, time

run_time_suffix = datetime.now()
run_time_suffix = run_time_suffix.strftime("%d%m%Y%H%M%S")

HOST = os.getenv("CDSW_API_URL").split(
    ":")[0] + "://" + os.getenv("CDSW_DOMAIN")
USERNAME = os.getenv("CDSW_PROJECT_URL").split(
    "/")[6]
API_KEY = os.getenv("CDSW_API_KEY")
PROJECT_NAME = os.getenv("CDSW_PROJECT")

### Instantiate API Wrapper
cml = CMLApi(HOST, USERNAME, API_KEY, PROJECT_NAME)

### Get User Details
user_details = cml.get_user({})
user_obj = {"id": user_details["id"], "username": "vdibia",
            "name": user_details["name"],
            "type": user_details["type"],
            "html_url": user_details["html_url"],
            "url": user_details["url"]
            }

# Get Project Details
project_details = cml.get_project({})
project_id = project_details["id"]

# Get Default Engine Details
default_engine_details = cml.default_engine({})
default_engine_image_id = default_engine_details["id"]

# Create Model
example_model_input = {"v": [-1.3598071336738,-0.0727811733098497,2.53634673796914,1.37815522427443,-0.338320769942518,0.462387777762292,0.239598554061257,0.0986979012610507,0.363786969611213,0.0907941719789316,-0.551599533260813,-0.617800855762348,-0.991389847235408,-0.311169353699879,1.46817697209427,-0.470400525259478,0.207971241929242,0.0257905801985591,0.403992960255733,0.251412098239705,-0.018306777944153,0.277837575558899,-0.110473910188767,0.0669280749146731,0.128539358273528,-0.189114843888824,0.133558376740387,-0.0210530534538215], 
                    "time": 0, "amount": 149.62}

create_model_params = {
    "projectId": project_id,
    "name": "Fraud Detection " + run_time_suffix,
    "description": "Fraud Detection",
    "visibility": "private",
    "targetFilePath": "2_fraud-model-deploy.py",
    "targetFunctionName": "predict",
    "engineImageId": default_engine_image_id,
    "kernel": "python3",
    "examples": [
        {
            "request": example_model_input,
            "response": {}
        }],
    "cpuMillicores": 1000,
    "memoryMb": 2048,
    "nvidiaGPUs": 0,
    "authEnabled": False,
    "replicationPolicy": {"type": "fixed", "numReplicas": 1},
    "environment": {}}

new_model_details = cml.create_model(create_model_params)
access_key = new_model_details["accessKey"]  # todo check for bad response
model_id = new_model_details["id"]

print("New model created with access key", access_key)

#Wait for the model to deploy.
is_deployed = False
while is_deployed == False:
  model = cml.get_model({"id": str(new_model_details["id"]), "latestModelDeployment": True, "latestModelBuild": True})
  if model["latestModelDeployment"]["status"] == 'deployed':
    print("Model is deployed")
    break
  else:
    print ("Deploying Model.....")
    time.sleep(10)
