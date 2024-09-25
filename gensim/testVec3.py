from gensim.models import Word2Vec
import numpy as np
import pandas as pd

# 데이터 로드
df = pd.read_excel('data4.xlsx')

# 사용자별로 그룹화하고, 각 그룹의 검색어를 리스트로 변환
grouped = df.groupby('name')['search'].apply(list)

# 모델 로드
w2v_model = Word2Vec.load('word2vec_model3.model')

user = 'cms8775'  # 사용자 ID

# 사용자가 검색한 단어들을 찾음
user_searches = grouped[grouped.index == user].values[0]

# 사용자가 검색한 각 단어의 벡터를 계산
user_vectors = [w2v_model.wv[word] for word in user_searches if word in w2v_model.wv.key_to_index]

# 사용자의 평균 벡터를 계산
user_vector = np.mean(user_vectors, axis=0)

# 사용자의 평균 벡터와 가장 유사한 벡터를 가진 단어들을 찾음
recommended_searches = w2v_model.wv.most_similar(positive=[user_vector], topn=5)

print(" === "+user+" 사용자에게 추천하는 검색어:")
for word, similarity in recommended_searches:
    print(f"- {word}: {similarity}")