import pickle

keyword = '귤'

# 모델 로드
with open('model.pkl', 'rb') as f:
    rules = pickle.load(f)

# 추천 시스템 구현
def recommend(fruit, top_n=5):
    recommendations = rules[rules['antecedents'].apply(lambda x: fruit in x)]
    # 지지도와 신뢰도의 곱을 기준으로 정렬
    recommendations['score'] = recommendations['support'] * recommendations['confidence']
    recommendations = recommendations.sort_values(by='score', ascending=False)
    unique_recommendations = []
    for consequents in recommendations['consequents']:
        for item in consequents:
            if item not in unique_recommendations and item != fruit:
                unique_recommendations.append(item)
            if len(unique_recommendations) == top_n:
                break
        if len(unique_recommendations) == top_n:
            break
    # 추천 과일의 개수가 부족할 경우 다른 과일을 추가
    all_fruits = set(rules['antecedents'].explode()).union(set(rules['consequents'].explode()))
    for fruit in all_fruits:
        if fruit not in unique_recommendations and fruit != keyword:
            unique_recommendations.append(fruit)
        if len(unique_recommendations) == top_n:
            break
    return unique_recommendations

# 예시로 '귤'을 입력받아 추천 과일 5개 출력
top_recommendations = recommend(keyword)
print(f"'{keyword}'을(를) 구매한 고객에게 추천하는 과일:")
for idx, rec in enumerate(top_recommendations, 1):
    print(f"{idx}. {rec}")