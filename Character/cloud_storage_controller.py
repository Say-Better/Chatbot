import os
import pandas as pd
from google.cloud import storage
import tempfile
from google.oauth2 import service_account
import Character.feature_extracting as llm
from datetime import datetime, timezone, timedelta

class CloudStorageManager:
    def __init__(self, model_name: str, parameters: dict):
        self.model_name = model_name
        self.parameters = parameters
        
        cd = service_account.Credentials.from_service_account_file(os.environ["GOOGLE_APPLICATION_CREDENTIALS"])
        self.client = storage.Client(credentials=cd)

    def load_user_character(self, user_id):
        bucket_name = 'user_characteristics'
        file_name = f'{user_id}_characteristics.txt'
        
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(file_name)
        
        if blob.exists():
            with tempfile.NamedTemporaryFile(mode='r+', delete=False, newline='') as temp_file:
                blob.download_to_filename(temp_file.name)
                with open(temp_file.name, 'r', encoding='utf-8') as file:
                    user_character = file.read()
                os.remove(temp_file.name)
        else:
            user_character = ""

        return user_character
    
    def load_user_sentences(self, user_id):
        bucket_name = 'user_sentence_data-user_id-sentence'
        file_name = f'{user_id}_sentence.csv'
        
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(file_name)
        
        if blob.exists():
            with tempfile.NamedTemporaryFile(mode='r+', delete=False, newline='') as temp_file:
                blob.download_to_filename(temp_file.name)
                user_sentences = pd.read_csv(temp_file.name)
                os.remove(temp_file.name)
        else:
            user_sentences = pd.DataFrame(columns=['time_stamp', 'user_sentence', 'chatbot_output1', 'chatbot_output2'])

        # 현재 시간과 30분 전 시간 계산
        KST = timezone(timedelta(hours=9))
        now = datetime.now(KST)
        thirty_minutes_ago = now - timedelta(minutes=30)
        
        user_sentences['time_stamp'] = user_sentences['time_stamp'].apply(lambda x: x.replace('[', '').replace(']', ''))

        # time_stamp 컬럼을 datetime 형식으로 변환
        user_sentences['time_stamp'] = pd.to_datetime(user_sentences['time_stamp'])
        
        # tz-naive datetime 객체로 변환
        thirty_minutes_ago = thirty_minutes_ago.replace(tzinfo=None)
        
        # 최근 30분 이내의 데이터만 필터링
        recent_sentences = user_sentences[user_sentences['time_stamp'] >= thirty_minutes_ago]
        
        return recent_sentences

    def save_user_character(self, user_id, user_character):
        bucket_name = 'user_characteristics'
        file_name = f'{user_id}_characteristics.txt'
        
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(file_name)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, newline='') as temp_file:
            with open(temp_file.name, 'w', encoding='utf-8') as file:
                file.write(user_character)
            temp_file.flush()
            blob.upload_from_filename(temp_file.name)
            os.remove(temp_file.name)
    
    def summarize_and_update_characteristics(self, user_id):
        user_sentences = self.load_user_sentences(user_id)
        user_character = self.load_user_character(user_id)
        
        if user_sentences.empty and not user_character:
            return f"No user sentences or characteristics to summarize for user {user_id}."

        # 기존 특성과 최근 30분 데이터를 결합하여 표시
        combined_sentences = []

        if user_character:
            combined_sentences.append(f"사용자 특성: {user_character}")

        if len(user_sentences) > 10:
            for _, row in user_sentences.iterrows():
                sentence = row['user_sentence']
                chatbot_output1 = row['chatbot_output1']
                chatbot_output2 = row['chatbot_output2']
                
                combined_sentences.append(f"사용자 대화 데이터: {sentence}")
                combined_sentences.append(f"챗봇 응답1: {chatbot_output1}")
                combined_sentences.append(f"챗봇 응답2: {chatbot_output2}")

            all_sentences = ", ".join(combined_sentences)

            # 요약을 위해 LLM 호출
            summary_result = llm.feature_extraction(
                'ivory-partition-421911', 
                "chat-bison@002", 
                self.parameters["temperature"], 
                self.parameters["max_output_tokens"], 
                self.parameters["top_p"], 
                self.parameters["top_k"], 
                all_sentences, 
                "us-central1"
            )

            summary = summary_result.text.strip()

            # 요약된 결과를 저장
            self.save_user_character(user_id, summary)
        
            return f"User {user_id} characteristics updated successfully."
        
        return f"Do not update User {user_id} characteristics."

    def update_all_users(self):
        bucket_name = 'user_sentence_data-user_id-sentence'
        bucket = self.client.bucket(bucket_name)
        
        blobs = list(bucket.list_blobs())
        user_ids = set(blob.name.split('_')[0] for blob in blobs if '_sentence.csv' in blob.name)
        
        results = []
        for user_id in user_ids:
            result = self.summarize_and_update_characteristics(user_id)
            results.append(result)
        
        return results

# 예시 사용법
if __name__ == "__main__":
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./ivory-partition-421911-1a720f1b0352.json"
    parameters = {
        "temperature": 0.2,
        "max_output_tokens": 256,
        "top_p": 0.95,
        "top_k": 40,
    }
    manager = CloudStorageManager(model_name="chat-bison@002", parameters=parameters)
    results = manager.update_all_users()
    for result in results:
        print(result)
