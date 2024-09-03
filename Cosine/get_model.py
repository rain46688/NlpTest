import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import pickle

# 1. 데이터 로드
data = pd.read_excel('data4.xlsx', header=None, names=['user', 'search'])

# 2. 데이터 전처리
# 각 유저가 검색한 키워드를 리스트로 변환
user_searches = data.groupby('user')['search'].apply(lambda x: ','.join(x)).reset_index()
user_searches['search'] = user_searches['search'].apply(lambda x: x.split(','))

# 3. 상관 관계 분석
# 키워드 간의 상관 관계를 계산하기 위해 유저-키워드 매트릭스를 만듭니다.
vectorizer = CountVectorizer(tokenizer=lambda x: x, lowercase=False)
search_matrix = vectorizer.fit_transform(user_searches['search'])

# 키워드 간의 코사인 유사도를 계산합니다.
cosine_sim = cosine_similarity(search_matrix.T)

# 키워드 인덱스와 이름을 매핑합니다.
keywords = vectorizer.get_feature_names_out()
keyword_index = {keyword: idx for idx, keyword in enumerate(keywords)}

# 4. 모델 저장
model = {
    'cosine_sim': cosine_sim,
    'keywords': keywords,
    'keyword_index': keyword_index
}

with open('keyword_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("모델이 성공적으로 저장되었습니다.")