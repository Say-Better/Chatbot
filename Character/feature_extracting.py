import vertexai
from vertexai.language_models import ChatModel, InputOutputTextPair
from vertexai.preview.language_models import TextGenerationModel

def feature_extraction(
    project_id: str,
    model_name: str,
    temperature: float,
    max_decode_steps: int,
    top_p: float,
    top_k: int,
    content: str,
    location: str = "us-central1",
    ) :
    """Predict using a Large Language Model."""
    vertexai.init(project=project_id, location=location)
    model = ChatModel.from_pretrained(model_name)

    parameters = {
            "temperature": 0.2,  # Temperature controls the degree of randomness in token selection.
            "max_output_tokens": 256,  # Token limit determines the maximum amount of text output.
            "top_p": 0.95,  # Tokens are selected from most probable to least until the sum of their probabilities equals the top_p value.
            "top_k": 40,  # A top_k of 1 means the selected token is the most probable among all tokens.
        }

    chat =  model.start_chat(
        context="너는 내가 보낸 사용자의 특성과 대화 데이터를 바탕으로 유저의 특성을 간단하게 업데이트 해줘야해 특수문자를 쓰지말고 txt형식으로만 output을 내보내줘",
        examples=[
            InputOutputTextPair(
                input_text="사용자 특성: 성별: 여성, 나이: 30대, 취미: 음악 감상,\
                사용자 대화 데이터: 오늘 날씨가 어때?, 챗봇 응답1: 오늘 날씨는 맑고 매우 쾌적해., 챗봇 응답2: 너는 어떤 날씨를 좋아해?,\
                사용자 대화 데이터: 나는 맑은 날씨가 좋아, 챗봇 응답1: 다행이다., 챗봇 응답2: 날씨가 맑으면 기분도 좋은 법이지ㅎ." ,
                output_text="사용자 특성: 성별: 여성, 나이: 30대, 취미: 음악 감상, 좋아하는 날씨: 맑은 날씨",
            ),
        ],
    )
    response = chat.send_message(
        content, **parameters
    )
    print(f"Response from Model: {response.text}")

    return response