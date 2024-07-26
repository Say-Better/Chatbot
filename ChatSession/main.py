import os
import numpy as np
import pandas as pd
import vertexai
from vertexai.language_models import ChatModel, InputOutputTextPair
from vertexai.preview.language_models import TextGenerationModel
import large_language_model_single_turn as llm
from google.cloud import storage
import tempfile
from google.oauth2 import service_account
import functions_framework
import ChatSession.cloud_storage_manager as csm

@functions_framework.http
def start(request):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./ivory-partition-421911-1a720f1b0352.json"

    cd = service_account.Credentials.from_service_account_file(os.environ["GOOGLE_APPLICATION_CREDENTIALS"])
    client = storage.Client(credentials=cd)
    # 'name' 키에 해당하는 JSON 데이터에 접근
    request_data = request.json

    sentence = request_data.get('sentence', '')
    user_id = request_data.get('user_id', '')

    parameters = {
        "temperature": 0.2,
        "max_output_tokens": 256,
        "top_p": 0.95,
        "top_k": 40,
    }
    
    headers = {
        'Content-Type': 'application/json'
    }

    cloud_storage_manager = csm.cloud_storage_manager(model_name="chat-bison@002", parameters=parameters, user_id=user_id)

    results = cloud_storage_manager.send_message(sentence)

    return (results, 200, headers)
