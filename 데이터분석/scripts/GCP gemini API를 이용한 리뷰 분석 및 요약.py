### 필요한 함수 임폴트
import os
from dotenv import load_dotenv
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core import Settings, Document, VectorStoreIndex
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
import tqdm
import pandas as pd
import json
import pickle
from dotenv import load_dotenv
import google.generativeai as generativeai
import random
import time
import re

### 리뷰 통합
file_path = '데이터분석/input/통합데이터(간단한_텍스트_전처리)_sample.csv'

df = pd.read_csv(file_path)

# groupby를 이용하여 리뷰 묶기
df_grouped = (
    df.groupby('고유번호')
      .agg({
          '제품명': 'first',
          '상세정보': 'first',
          '카테고리': 'first',
          '리뷰': lambda x: ' / '.join(x.fillna('').astype(str))
      })
      .reset_index()
)

df_grouped.to_csv('outputs/1.gemini_리뷰통합_result.csv',index=False)


### df_grouped 데이터 프레임 임폴트

file_path = '데이터분석/input/리뷰통합_sample.csv'

df_grouped = pd.read_csv(file_path)

### .env file 로드
load_dotenv('.env')

### gemini key 불러오기기
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
generativeai.configure(api_key=GOOGLE_API_KEY)

# 리뷰 텍스트 생성
review_text = df_grouped.loc[:, '리뷰'].to_list()

### 감성 분석 및 근거가 되는 keyword 추출 함수 정의  --> 12시간 걸림
def analyze_review(text_input):
    # prompt 생성
    prompt = f"""
    다음 텍스트는 건강기능식품에 관한 소비자 리뷰입니다. 제품 하나당 모든 리뷰가 '/'로 구분되어져 포함되어 있습니다.
    해당 리뷰의 내용에서 긍정인 내용, 부정인 내용, 그리고 중립인 내용을 분류를 하고, 각 감성별로 리뷰를 요약하여 JSON 형식으로 제시해 주세요.
    각 감성별 리뷰 요약은 맥락이 비슷한 것만 남기고 요약을 하여, 각 감성별 리뷰의 글자 수가 300자를 넘지 않도록 요약해 주세요.
    그리고 리뷰 요약할 때 없는 내용을 창조하면 안되고, 리뷰 내용에 100% 충실하게 요약해야 해야 합니다.
    JSON 형식만 출력하세요. 아래 형식 외의 설명 문구는 절대 쓰지 마세요.

    텍스트: {text_input}

    출력 형식 예시:
    {{
      "정확도": 85,
      "감성별 리뷰 요약": {{
        "긍정": ["긍정 리뷰 요약"],
        "부정": ["부정 리뷰 요약"],
        "중립": ["중립 리뷰 요약"]
      }}
    }}
    """


    try:
        # 텍스트 분석 결과 생성
        model = generativeai.GenerativeModel("gemini-2.5-pro")

        # generation_config 객체를 생성하여 temperature를 설정합니다.
        # temperature 값은 0.0 (가장 보수적)에서 1.0 (가장 창의적) 사이로 조절할 수 있습니다.
        generation_config = generativeai.types.GenerationConfig(
            temperature=0.0,  # 원하는 temperature 값으로 변경하세요 (예: 0.2, 0.5, 0.9 등)
            top_p=0.9,
            top_k=50
        )

        response = model.generate_content(
            contents=[prompt],
            generation_config=generation_config # 여기에 generation_config를 전달합니다.
        )


        result_str = response.text
        if result_str:
            # print(f'결과 : {result_str}')

            # JSON 부분만 추출
            match = re.search(r"\{[\s\S]*\}", result_str)
            if match:
                json_text = match.group(0).strip()
                try:
                    parsed_json = json.loads(json_text)
                    return parsed_json
                except json.JSONDecodeError:
                    print("JSON 디코딩 실패. 추출된 내용:", json_text)
                    return None
            else:
                print("JSON 패턴을 찾지 못했습니다.")
                return None


    except Exception as e:
        print(f'API 호출 오류 : {e}')
        return None



### 전체 텍스트를 처리하는 함수 정의 (tqdm 적용)
def process_multiple_texts(text_list):
    results = {}
    for i, text_input in enumerate(tqdm(text_list, desc="리뷰 처리 진행률")):
        retry_count = 0
        max_retries = 5  # 최대 재시도 횟수
        wait_time = 20  # 초기 대기 시간 (초)

        while retry_count < max_retries:
            print(f"텍스트 {i+1} 처리 시도 {retry_count + 1}...")
            result = analyze_review(text_input)
            if result:
                # 성공 조건을 "올바른 딕셔너리(JSON 파싱 성공)"로 한정
                if isinstance(result, dict):
                    results[f"텍스트 {i+1}"] = result
                    break  # 성공 시 루프 종료
                else:
                    retry_count += 1
                    wait_time = wait_time * 2 + random.uniform(0, 1)
                    print(f"API 요청 또는 JSON 파싱 실패. {wait_time:.2f}초 후 재시도...")
                    time.sleep(wait_time)

        else:  # 최대 재시도 횟수 초과 시
            results[f"텍스트 {i+1}"] = "감정 키워드 추출 실패 (최대 재시도 횟수 초과)"

    return results


# 여러 텍스트 처리
all_results = process_multiple_texts(review_text)

# 최종 결과를 dict 자료 구조로 변환
with open('review_analysis.pkl', 'wb') as fw:
    pickle.dump(all_results, fw)

# 저장된 결과를 다시 불러오기
with open('review_analysis.pkl', 'rb') as fr:
    loaded_data = pickle.load(fr)

# print(f'리뷰 데이터 분석의 결과 : \n{loaded_data}')

### DataFrame에 붙이기

df_grouped["리뷰_분석"] = list(loaded_data.values())


### 감성별로 리뷰 따로 떼서 열 만들기 (임베딩 용)

# 긍정/부정/중립 리뷰 요약만 새 열로 추가
df_grouped['긍정_리뷰'] = df_grouped['리뷰_분석'].apply(lambda x: x['감성별 리뷰 요약']['긍정'] if isinstance(x, dict) else None)
df_grouped['부정_리뷰'] = df_grouped['리뷰_분석'].apply(lambda x: x['감성별 리뷰 요약']['부정'] if isinstance(x, dict) else None)
df_grouped['중립_리뷰'] = df_grouped['리뷰_분석'].apply(lambda x: x['감성별 리뷰 요약']['중립'] if isinstance(x, dict) else None)

# df_grouped에서 '리뷰' 열 삭제

df_grouped.drop(columns='리뷰', inplace=True)

# csv 저장

df_grouped.to_csv('outputs/2.gemini_리뷰분석_result.csv', index=False)

# csv 확인

import pandas as pd

file_path = '데이터분석/output/6.리뷰분석(감성_리뷰만 있는 버전)_sample.csv'

df = pd.read_csv(file_path)

print(df)
