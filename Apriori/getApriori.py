import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import pickle

# 1. 데이터 로드
data = pd.read_excel('data3.xlsx')

# 2. 데이터 전처리
# 각 구매 내역을 리스트로 변환
transactions = data['name'].apply(lambda x: x.split(','))

# 고유한 과일 목록 추출
all_fruits = set()
for transaction in transactions:
    all_fruits.update(transaction)

# 과일 목록을 이용해 원-핫 인코딩
encoded_data = []
for transaction in transactions:
    encoded_row = {fruit: (fruit in transaction) for fruit in all_fruits}
    encoded_data.append(encoded_row)

df_encoded = pd.DataFrame(encoded_data)

# 3. 연관 규칙 학습
# 최소 지지도 0.1로 설정하여 빈번한 항목 집합 찾기
frequent_itemsets = apriori(df_encoded, min_support=0.1, use_colnames=True)

# 신뢰도 0.5로 설정하여 연관 규칙 생성
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

# 모델 저장
with open('model.pkl', 'wb') as f:
    pickle.dump(rules, f)