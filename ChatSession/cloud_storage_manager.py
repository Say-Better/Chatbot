import os
import pandas as pd
from vertexai.language_models import ChatModel, InputOutputTextPair
from vertexai.preview.language_models import TextGenerationModel
from google.cloud import storage
import tempfile
from google.oauth2 import service_account
import large_language_model_single_turn as llm
from datetime import date, datetime, timezone, timedelta
import json


class cloud_storage_manager:
    def __init__(self, model_name: str, parameters: dict, user_id: str):
        self.model_name = model_name
        self.parameters = parameters
        self.user_id = user_id
        
        cd = service_account.Credentials.from_service_account_file(os.environ["GOOGLE_APPLICATION_CREDENTIALS"])
        self.client = storage.Client(credentials=cd)

    def load_user_character(self):
        bucket_name = 'user_characteristics'
        file_name = f'{self.user_id}_characteristics.csv'
        
        # Check if the file exists in the bucket
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(file_name)
        
        if blob.exists():
            # Download the existing file
            with tempfile.NamedTemporaryFile(mode='r+', delete=False, newline='') as temp_file:
                blob.download_to_filename(temp_file.name)
                # Load the file into a DataFrame
                user_character = pd.read_txt(temp_file.name)
                os.remove(temp_file.name)
        else:
            user_character = ''

        return user_character
    
    def send_message(self, content: str):
        KST = timezone(timedelta(hours=9))
        time_record = datetime.now(KST)
        _day = str(time_record)[:10]
        _time = str(time_record.time())[:8]

        time_stamp = _day + ' ' + _time

        user_character = self.load_user_character()

        if user_character != '':
            sentence = '사용자의 특성: {0}, 사용자의 입력: {1}'.format(user_character, content)
        else:
            sentence = '사용자의 입력: {0}'.format(content)

        results = llm.predict_large_language_model_sample(
            'ivory-partition-421911', "chat-bison@002", self.parameters["temperature"], self.parameters["max_output_tokens"], self.parameters["top_p"], self.parameters["top_k"], sentence, "us-central1"
        )

        multi_parts = (results.text).strip('[]').split('], [')

        if len(multi_parts) == 3:
            answer, question, multi_score = multi_parts
            answer = answer.replace('[', '')
            multi_score = int(multi_score)
        else:
            return (json.dumps({'answer1':"다시 한번만 설명해줄래?", 'answer2':"조금만 구체적으로 설명해주면 좋겠어!", 'score':10}))

        if multi_score < 80:
            # 정상적인 대화 데이터
            bucket_name = 'user_sentence_data-user_id-sentence'
            file_name = self.user_id + '_sentence.csv'

            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(file_name)

            if not blob.exists():
                with tempfile.NamedTemporaryFile(mode='w', delete=False, newline='') as temp_file:
                    df = pd.DataFrame({'time_stamp': [time_stamp], 'user_sentence': [sentence], 'chatbot_output1': [answer], 'chatbot_output2': [question]})
                    df.to_csv(temp_file.name, index=False)
                    temp_file.flush()
                    blob.upload_from_filename(temp_file.name)
                    os.remove(temp_file.name)
            else:
                with tempfile.NamedTemporaryFile(mode='r+', delete=False, newline='') as temp_file:
                    blob.download_to_filename(temp_file.name)
                    df = pd.read_csv(temp_file.name)
                    new_row = {'time_stamp': [time_stamp], 'user_sentence': [sentence], 'chatbot_output1': [answer], 'chatbot_output2': [question]}
                    new_row_df = pd.DataFrame([new_row])
                    df = pd.concat([df, new_row_df], ignore_index=True)
                    df.to_csv(temp_file.name, index=False)
                    temp_file.flush()
                    blob.upload_from_filename(temp_file.name)
                    os.remove(temp_file.name)
        else:
            # Save to fail_sentence
            bucket_name = 'user_sentence_data-user_id-fail_sentence'
            file_name = self.user_id + '_fail_sentence.csv'

            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(file_name)

            if not blob.exists():
                with tempfile.NamedTemporaryFile(mode='w', delete=False, newline='') as temp_file:
                    df = pd.DataFrame({'time_stamp': [time_stamp], 'user_sentence': [sentence], 'chatbot_output1': [answer], 'chatbot_output2': [question]})
                    df.to_csv(temp_file.name, index=False)
                    temp_file.flush()
                    blob.upload_from_filename(temp_file.name)
                    os.remove(temp_file.name)
            else:
                with tempfile.NamedTemporaryFile(mode='r+', delete=False, newline='') as temp_file:
                    blob.download_to_filename(temp_file.name)
                    df = pd.read_csv(temp_file.name)
                    new_row = {'time_stamp': [time_stamp], 'user_sentence': [sentence], 'chatbot_output1': [answer], 'chatbot_output2': [question]}
                    new_row_df = pd.DataFrame([new_row])
                    df = pd.concat([df, new_row_df], ignore_index=True)
                    df.to_csv(temp_file.name, index=False)
                    temp_file.flush()
                    blob.upload_from_filename(temp_file.name)
                    os.remove(temp_file.name)
            return (json.dumps({'answer1':"대화 맥락이 이상한 것 같아요. 다시 입력해줘.", 'answer2':"조금만 구체적으로 설명해주면 좋겠어!", 'score':100}))

        results = json.dumps({'answer1': answer, 'answer2': question, 'score': multi_score})
        print(results)
        return results
