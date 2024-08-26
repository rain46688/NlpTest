import pandas as pd
from gensim.models import FastText
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 모델 로드
loaded_model = FastText.load('fasttext_model.model')
print("FastText 모델이 로드되었습니다.")

# 특정 단어의 벡터 확인 및 유사 단어 검색
keyword = '귤'  # 확인할 단어를 지정하세요

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

# FastText 모델에서 단어 벡터 가져오기
word_vectors = loaded_model.wv
vocabs = list(word_vectors.index_to_key)  # vocab 대신 사용
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
