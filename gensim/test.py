import gensim
from gensim.models import LdaModel

# 검색 키워드
keyword = "고양이"

# 모델과 사전 불러오기
lda_model = LdaModel.load('lda_model.gensim')

# 사전은 모델에 저장된 것을 가져옵니다.
dictionary = lda_model.id2word

# 단어의 벡터를 찾기 위해 모든 단어의 토픽 분포를 구합니다.
def get_topic_distribution(word):
    if word not in dictionary.token2id:
        print(f"단어 '{word}'가 사전에 없습니다.")
        return None
    
    word_id = dictionary.token2id[word]
    word_vec = dictionary.doc2bow([word])
    
    # 단어의 토픽 분포 구하기
    topic_distribution = lda_model[word_vec]
    return topic_distribution

# 유사한 단어 찾기
def find_similar_words(target_word, num_similar=3):
    topic_distribution = get_topic_distribution(target_word)
    if topic_distribution is None:
        return
    
    # 각 단어의 유사도를 저장할 리스트
    word_similarities = []
    
    # 모든 단어를 반복하면서 유사도 측정
    for word in dictionary.token2id:
        if word == target_word:
            continue
        word_vec = dictionary.doc2bow([word])
        word_distribution = lda_model[word_vec]
        
        # 유사도 계산: 두 벡터의 내적을 사용
        similarity = sum(topic_prob * word_prob for (topic_id, topic_prob), (_, word_prob) in zip(topic_distribution, word_distribution))
        word_similarities.append((word, similarity))
    
    # 유사도 높은 단어 정렬
    word_similarities.sort(key=lambda x: x[1], reverse=True)
    
    # 결과 출력
    print(f" === 키워드 '{target_word}' 유사한 단어 === ")
    for word, similarity in word_similarities[:num_similar]:
        print(f"단어 : {word}, 유사도 : {similarity:.4f}")

# 사용 예시
find_similar_words(keyword, num_similar=3)
