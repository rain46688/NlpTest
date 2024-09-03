import pickle

keyword = '귤'

# 1. 모델 로드
with open('keyword_model.pkl', 'rb') as f:
    model = pickle.load(f)

cosine_sim = model['cosine_sim']
keywords = model['keywords']
keyword_index = model['keyword_index']

# 2. 추천 함수 정의
def recommend_keywords(keyword, top_n=6):
    if keyword not in keyword_index:
        return []
    idx = keyword_index[keyword]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    recommended_keywords = [(keywords[i], score) for i, score in sim_scores[1:top_n+1]]
    return recommended_keywords

# 3. 테스트
recommended = recommend_keywords(keyword)
print(f"'{keyword}'을(를) 검색한 사람에게 추천할 키워드:")
for rank, (word, score) in enumerate(recommended, start=1):
    print(f"{rank}. {word} (연관도: {score:.2f})")