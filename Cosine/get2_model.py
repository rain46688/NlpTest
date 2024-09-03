import pandas as pd
from collections import defaultdict
import pickle

# 1. 데이터 로드
data = pd.read_excel('data4.xlsx', header=None, names=['user', 'search'])

# 2. 데이터 전처리
# 각 유저가 검색한 키워드를 리스트로 변환
user_searches = data.groupby('user')['search'].apply(lambda x: ','.join(x)).reset_index()
user_searches['search'] = user_searches['search'].apply(lambda x: x.split(','))

# 3. 빈도수 계산
keyword_freq = defaultdict(lambda: defaultdict(int))

for searches in user_searches['search']:
    for i in range(len(searches)):
        for j in range(i + 1, len(searches)):
            keyword_freq[searches[i]][searches[j]] += 1
            keyword_freq[searches[j]][searches[i]] += 1

# defaultdict를 일반 딕셔너리로 변환
keyword_freq = {k: dict(v) for k, v in keyword_freq.items()}

# 4. 모델 저장
with open('keyword_freq_model.pkl', 'wb') as f:
    pickle.dump(keyword_freq, f)

print("모델이 성공적으로 저장되었습니다.")