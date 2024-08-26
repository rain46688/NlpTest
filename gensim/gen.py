# https://github.com/piskvorky/gensim
import gensim
from gensim import corpora
from gensim.models import LdaModel
import os
import pandas as pd
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

# 데이터 준비
# 엑셀 파일 읽기
df = pd.read_excel('data.xlsx')
documents = df['name'].apply(lambda x: x.split()).tolist()

# 단어 사전 만들기
dictionary = corpora.Dictionary(documents)

# 문서를 벡터 형태로 변환
corpus = [dictionary.doc2bow(doc) for doc in documents]

# 기존 모델 삭제 (필요한 경우)
if os.path.exists('lda_model.gensim'):
    os.remove('lda_model.gensim')

# LDA 모델 학습
lda_model = LdaModel(corpus, num_topics=2, id2word=dictionary, passes=15)

# 모델 저장
lda_model.save('lda_model.gensim')

# 학습된 토픽 출력
print("학습된 토픽:")
topics = lda_model.print_topics(num_words=3)
for topic in topics:
    print(topic)

# 각 문서의 토픽 분포 출력
print("\n각 문서의 토픽 분포:")
for i, doc_bow in enumerate(corpus):
    doc_topics = lda_model[doc_bow]
    print(f"문서 {i}의 토픽 분포: {doc_topics}")
