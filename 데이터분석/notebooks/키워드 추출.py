### 필요한 라이브러리 임폴트

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

### 감성 분석+리뷰 요약 데이터프레임 불러오기

df = pd.read_csv(''데이터분석/한글+영어 최종본_sample.csv'')

### 카테고리 열에서 내가 담당한 '아연'과 '오메가3'만 불러오기

condition = df.loc[:,'카테고리'].isin(['아연','오메가3'])

df1 = df.loc[condition,:]

# 아연 (긍정, 부정, 중립)

con1 = (df.loc[:,'카테고리']=='아연')&(df.loc[:,'sentiment']=='positive')
zinc_pos = df.loc[con1,:]

con2 = (df.loc[:,'카테고리']=='아연')&(df.loc[:,'sentiment']=='negative')
zinc_neg = df.loc[con2,:]

con3 = (df.loc[:,'카테고리']=='아연')&(df.loc[:,'sentiment']=='neutral')
zinc_neu = df.loc[con3,:]

# 오메가3 (긍정, 부정, 중립)

con4 = (df.loc[:,'카테고리']=='오메가3')&(df.loc[:,'sentiment']=='positive')
omega3_pos = df.loc[con4,:]

con5 = (df.loc[:,'카테고리']=='오메가3')&(df.loc[:,'sentiment']=='negative')
omega3_neg = df.loc[con5,:]

con6 = (df.loc[:,'카테고리']=='오메가3')&(df.loc[:,'sentiment']=='neutral')
omega3_neu = df.loc[con6,:]


# 키워드 추출 함수 생성
def extract_keywords(df, text_col, top_n=50):
    # 리뷰 데이터프레임
    reviews = df.loc[:,text_col]

    # TF-IDF 벡터라이저 객체 생성
    tfidf_vectorizer = TfidfVectorizer()

    # 리뷰에 fit_transform 적용
    # 결과: tfidf_matrix → (리뷰 수 x 단어 수) 크기의 희소 행렬
    tfidf_matrix = tfidf_vectorizer.fit_transform(reviews)

    # 단어와 TF-IDF 점수 매핑
    feature_names = tfidf_vectorizer.get_feature_names_out()
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

    # 모든 리뷰의 TF-IDF 점수를 합산
    # 이 과정이 전체 문서 집합에서 가장 중요한 키워드를 찾는 핵심
    total_tfidf = tfidf_df.sum(axis=0)

    # 합산된 점수를 기준으로 내림차순 정렬하여 상위 50개 키워드 추출
    top_keywords = total_tfidf.sort_values(ascending=False)

    # 결과
    return top_keywords

datasets = {
    "zinc_pos": zinc_pos,
    "zinc_neg": zinc_neg,
    "zinc_neu": zinc_neu,
    "omega3_pos": omega3_pos,
    "omega3_neg": omega3_neg,
    "omega3_neu": omega3_neu
}

results = {}
for name, df in datasets.items():
    results[name] = extract_keywords(df, 'summary')

# 출력
for name, keywords in results.items():
    print(f"\n=== {name} ===")
    print(keywords)
