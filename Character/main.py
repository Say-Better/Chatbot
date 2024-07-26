import os
import numpy as np
import pandas as pd
import vertexai
from google.cloud import storage
import tempfile
from google.oauth2 import service_account
import functions_framework
import Character.cloud_storage_controller as csc

@functions_framework.http
def start(request):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./ivory-partition-421911-1a720f1b0352.json"

    parameters = {
        "temperature": 0.2,
        "max_output_tokens": 256,
        "top_p": 0.95,
        "top_k": 40,
    }
    

    cloud_storage_controller = csc.CloudStorageManager(model_name="text-bison@002", parameters=parameters)

    results = cloud_storage_controller.update_all_users()

    return (results, 200)
