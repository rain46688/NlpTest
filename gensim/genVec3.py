import os
import pandas as pd
from gensim.models import Word2Vec

# 데이터 로드
df = pd.read_excel('data4.xlsx')

# 사용자별로 그룹화하고, 각 그룹의 검색어를 리스트로 변환
grouped = df.groupby('name')['search'].apply(list)

# Word2Vec 모델 학습
w2v_model = Word2Vec(
    sentences=grouped,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    epochs=100
)

# 모델 저장
w2v_model.save('word2vec_model3.model')