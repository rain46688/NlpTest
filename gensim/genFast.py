import os
import pandas as pd
from gensim.models import FastText

# 데이터 준비
# 엑셀 파일 읽기
df = pd.read_excel('data3.xlsx')

# 전처리: 문자열을 쉼표로 분리하고 토큰화
documents = df['name'].apply(lambda x: str(x).lower().split(',')).tolist()

print("토큰화된 문서들:")
print(documents)

# 기존 모델 삭제 (필요한 경우)
if os.path.exists('fasttext_model.model'):
    os.remove('fasttext_model.model')

# FastText 모델 학습
fasttext_model = FastText(
    sentences=documents,
    vector_size=100,     # 벡터 크기
    window=5,           # 컨텍스트 윈도우 크기
    min_count=5,         # 최소 단어 빈도수
    workers=4,           # 병렬 처리 스레드 수
    epochs=100,          # 학습 에폭 수
    sg=1,                # skip-gram 모델 사용
    negative=10          # 부정 샘플링 수
)

print(fasttext_model.wv.vectors.shape)  # 학습된 단어 벡터의 크기

# 모델 저장
fasttext_model.save('fasttext_model.model')
print("FastText 모델이 'fasttext_model.model'로 저장되었습니다.")
