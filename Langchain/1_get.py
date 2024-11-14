import os
import pandas as pd
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_teddynote import logging
import pickle

# API 키 정보 로드
load_dotenv()

# 프로젝트 이름을 입력합니다.
logging.langsmith("fruits_embedding")

# OpenAI의 "text-embedding-3-small" 모델을 사용하여 임베딩을 생성합니다.
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# print(f"[API KEY]\n{os.environ['OPENAI_API_KEY']}")

# 엑셀 파일 읽기
df = pd.read_excel('data4.xlsx')

# 전처리: 문자열을 쉼표로 분리하고 토큰화
grouped = df.groupby('name')['search'].apply(lambda x: [i.strip().lower() for sublist in x for i in sublist.split(',')]).reset_index()

# 토큰화된 문서들을 리스트로 변환
documents = grouped['search'].tolist()

print(documents[0])

# 각 문서에 대해 벡터 생성
document_vectors = embeddings.embed_documents(documents[0])

# You exceeded your current quota, please check your plan and billing details
# 돈내야함

