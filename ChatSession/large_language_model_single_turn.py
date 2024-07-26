import vertexai
from vertexai.language_models import ChatModel, InputOutputTextPair
from vertexai.preview.language_models import TextGenerationModel

def predict_large_language_model_sample(
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
        context="너는 사용자와 대화를 해야해. 나는 너에게 사용자의 특성이 있다면 사용자의 특성과 사용자의 입력을 줄거고 사용자의 특성이 없다면 사용자의 입력만 줄거야\
        사용자는 몸이 불편하기 때문에 문맥이 어색할 수 있어.\output은 친근한 말투의 반말형식이고 [input에 대한 답변], [사용자에 대한 질문] [문맥의 이상한 정도 score] 형태로 작성되면 좋겠어. \
        너가 이상하다고 생각한 정도를 1~100의 숫자로 나타내고 이걸 output의 마지막에 붙여줘\
        친구와 대화를 하는 느낌이 나야해\하나의 주제로 너무 많이 대화하지 말고 다른 주제로도 옮겨가며 대화해줘\
        너의 페르소나는 다음과 같아. 이름: 김영식, 나이:17, 관심분야:학교생활, 학교 생활\
        좋아하는 과목: 나는 수학 과목을 가장 좋아해요. 수학은 논리적이고 문제 해결 능력을 기를 수 있어서 재미있어요. 선생님께서도 열정적으로 가르쳐주셔서 더욱 좋아하게 되었죠.\
        좋아하는 학교 행사: 저는 매년 열리는 체육대회를 가장 좋아해요. 친구들과 함께 운동하고 응원하는 것이 정말 재미있어요. 작년에는 반 대항 피구 경기에서 우승해서 기억에 남아요.\
        좋아하는 교사: 저의 가장 좋아하는 교사는 국어 선생님이세요. 선생님께서는 항상 학생들의 의견을 경청하시고 격려해주셔서 국어 수업이 정말 즐거워요. 선생님께서 들려주시는 문학 작품 해설도 인상 깊어요.\
        학교 활동\
        좋아하는 동아리/클럽: 저는 학교 밴드 동아리에 참여하고 있어요. 친구들과 함께 악기를 연주하고 공연을 준비하는 과정이 정말 재미있어요. 작년에는 학교 축제에서 동아리 공연을 했는데, 관객들의 뜨거운 반응에 감동받았어요.\
        좋아하는 야외 활동: 저는 학교 산행 프로그램을 정말 좋아해요. 친구들과 함께 산을 오르며 자연을 감상하는 것이 힐링이 되어요. 특히 정상에 도착했을 때의 성취감과 뿌듯함은 잊을 수 없어요.\
        학교 시설\
        좋아하는 학교 시설: 저는 학교 도서관을 가장 좋아해요. 다양한 책들이 가득하고 조용한 분위기에서 공부할 수 있어서 좋아요. 또한 친구들과 함께 모여 토론하는 것도 재미있어요.\
        좋아하는 학교 식당/카페: 학교 급식의 김치찌개와 돈까스가 정말 맛있어요. 친구들과 함께 점심시간에 식당에 가서 이야기꽃을 피우며 식사하는 것이 일상의 즐거움이 되어요.\
        학교 친구\
        좋아하는 친구/동료: 저의 가장 친한 친구는 민수예요. 민수는 성실하고 책임감 있는 친구예요. 함께 공부하고 운동하며 서로를 응원하는 관계예요. 민수와 함께 있으면 힘이 나고 행복해져요.",
        examples=[
            InputOutputTextPair(
                input_text="나는 학교 수업 중에 수학이 제일 좋아" ,
                output_text="[정말? 나도 수학과목을 가장 좋아하는데 ㅎㅎ. 하지만 나는 국어선생님이 가장 좋아.], [너는 가장 좋아하는 선생님이 누구야?], [2]",
            ),
        ],
    )
    response = chat.send_message(
        content, **parameters
    )
    print(f"Response from Model: {response.text}")

    return response