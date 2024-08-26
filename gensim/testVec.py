import os
import numpy as np
from gensim.models import Word2Vec
from scipy import spatial
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 모델 로드
loaded_model = Word2Vec.load('word2vec_model.model')
print("Word2Vec 모델이 로드되었습니다.")

# 특정 단어의 벡터 확인 및 유사 단어 검색
keyword = '귤'  # 확인할 단어를 지정하세요

# print(loaded_model.wv.most_similar(positive=[keyword], topn=10))

# =================================================================================================

if keyword in loaded_model.wv:
    print(f"\n단어 '{keyword}'의 벡터 표현:")
    print(loaded_model.wv[keyword])

    print(f"\n'{keyword}'와 유사한 단어들:")
    similar_words = loaded_model.wv.most_similar(keyword)
    for similar_word, similarity in similar_words:
        print(f"{similar_word}: {similarity}")
else:
    print(f"'{keyword}'는 모델의 어휘에 포함되지 않았습니다.")

# 한글 폰트 설정
rcParams['font.family'] = 'NanumGothic'  # 설치된 한글 폰트로 변경 가능

# 그래프 출력 함수 정의
def plot_2d_graph(vocabs, xs, ys):
    plt.figure(figsize=(8,6))
    plt.scatter(xs, ys, marker='o')
    for i, v in enumerate(vocabs):
        plt.annotate(v, xy=(xs[i], ys[i]))
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('Word Vectors Visualized in 2D')
    plt.grid(True)
    plt.show()

# Word2Vec 모델에서 단어 벡터 가져오기
word_vectors = loaded_model.wv
vocabs = list(word_vectors.index_to_key)  # 이전의 vocab 대신 사용
word_vectors_list = [word_vectors[v] for v in vocabs]

# PCA를 사용하여 단어 벡터 시각화
pca = PCA(n_components=2)
xys = pca.fit_transform(word_vectors_list)
xs = xys[:, 0]
ys = xys[:, 1]

print(pca)
print(xys)
print(xs)
print(ys)

# 그래프 출력
plot_2d_graph(vocabs, xs, ys)


# =================================================================================================

# 귤과 사과의 유사도 직접 계산
# if '사과' in loaded_model.wv and '귤' in loaded_model.wv:
#     similarity = loaded_model.wv.similarity('귤', '사과')
#     print(f"'귤'과 '사과'의 유사도: {similarity}")
# else:
#     print("모델에 '귤' 또는 '사과'가 포함되지 않았습니다.")

# # 데이터를 다시 로드 및 전처리
# df = pd.read_excel('data.xlsx')
# documents = df['name'].apply(lambda x: str(x).lower().split(',')).tolist()

# # 각 문서의 벡터 표현 계산 (문서 내 단어 벡터의 평균)
# document_vectors = []
# for doc in documents:
#     # 문서 내 존재하는 단어들의 벡터를 가져옴
#     valid_vectors = [loaded_model.wv[word] for word in doc if word in loaded_model.wv]
    
#     if valid_vectors:  # 유효한 단어 벡터가 있는 경우
#         vector = np.mean(valid_vectors, axis=0)
#     else:
#         vector = np.zeros(loaded_model.vector_size)  # 모든 단어가 어휘에 없으면 0으로 채움
    
#     document_vectors.append(vector)

# # 첫 번째 문서와 다른 문서 간의 유사도 계산 (예: 0번째 문서와 7번째 문서)
# similarity = 1 - spatial.distance.cosine(document_vectors[0], document_vectors[7])
# print(f"\n문서 0과 문서 7의 코사인 유사도: {similarity}")

# # 모든 문서 간의 유사도 매트릭스 계산
# similarity_matrix = np.zeros((len(document_vectors), len(document_vectors)))
# for i in range(len(document_vectors)):
#     for j in range(len(document_vectors)):
#         if i != j:
#             similarity = 1 - spatial.distance.cosine(document_vectors[i], document_vectors[j])
#             similarity_matrix[i][j] = similarity
#         else:
#             similarity_matrix[i][j] = 1.0  # 자기 자신과의 유사도는 1

# # 유사도 매트릭스 출력
# print("\n문서 유사도 매트릭스:")
# print(similarity_matrix)
