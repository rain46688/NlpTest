import pickle

keyword = '귤'

# 1. 모델 로드
with open('keyword_freq_model.pkl', 'rb') as f:
    keyword_freq = pickle.load(f)

# 2. 추천 함수 정의
def recommend_keywords(keyword, top_n=7):
    if keyword not in keyword_freq:
        return []
    freq_scores = keyword_freq[keyword].items()
    sorted_keywords = sorted(freq_scores, key=lambda x: x[1], reverse=True)
    recommended_keywords = [(word, score) for word, score in sorted_keywords[:top_n]]
    return recommended_keywords

# 3. 테스트
recommended = recommend_keywords(keyword)
print(f"'{keyword}'을(를) 검색한 사람에게 추천할 키워드:")
for rank, (word, score) in enumerate(recommended, start=1):
    print(f"{rank}. {word} (빈도수: {score})")