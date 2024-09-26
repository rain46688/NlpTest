from gensim.models import Word2Vec

# 모델 불러오기
w2v_model = Word2Vec.load('result_word2vec_model.model')
print(" === Word2Vec 모델이 로드되었습니다.")

# 키워드 설정
keyword = '장판'

# 키워드와 연관도가 높은 단어들 찾기
related_keywords = w2v_model.wv.most_similar(keyword)

print(f" === '{keyword}'과 연관도가 높은 키워드:")
for keyword, similarity in related_keywords:
    print(f"- {keyword} (유사도: {similarity})")