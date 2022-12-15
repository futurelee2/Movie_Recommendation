#단어를 백터화

import pandas as pd
from gensim.models import Word2Vec

review_word = pd.read_csv('./crawling_data/one_sentences.csv')
review_word.info()

one_sentence_reviews = list(review_word['reviews'])

cleaned_tokens = []
for setence in one_sentence_reviews: #형태소가 들어있는 리스트로 줘야함(형태소단위로 띄어쓰기 되어있음)
    token = setence.split()
    cleaned_tokens.append(token)

embedding_model = Word2Vec(cleaned_tokens, vector_size=100,
                           window=4, min_count=20,
                           workers=8, epochs=100, sg=1) #sg=1 중심에서 주변으로

embedding_model.save('./models/word2vec_movie_review.model')
print(list(embedding_model.wv.index_to_key))
print(len(embedding_model.wv.index_to_key))



#100으로 차원축소/ 4단어씩 잘라서 학습을 하겠다
# window =4는 convd1 레이어처럼 4단어만 보겟다
# min count : 20번 이하는 버린다(20넘어야 학습한다)/ word2vec 는 cpu여러개써서 병행처리 가능 (가진 cpu의 절반만 줘야함)
#bow: back of word (단어들의 종이봉투, 무작위)
# sg=0이면 cbow: 단어의 순서까지 고려, 오늘 저녁은 이태리,아시아(비슷한단어) 풍으로..
# 여름엔 수박, 여름휴가는 해수욕장(여름 수박 휴가 해수욕장: 비슷한단어)
#겨울 축에 겨울과 관련된 단어들이 겨울과 관계있으면 큰 값을 줌(의미차원=의미가 크면 큰값, 작으면 작은값)
#비슷한 의미를 가진 단어들이 가깝게 배치
#좌표(1,1)는 값일 뿐, 벡터(1,1)는 길이와 방향,크기(모든 값 제곱한뒤 루트)를 가짐
#word2vec 말뭉치들을 의미차원에 배치하고 각 단어들의 좌표를 만듦 > 유니크한 단어들의 개수만큼 차원(좌표)이 만들어지는 것 차원 축소 필요(학습을 하면서 의미공간에 좌표 배치됨)
#차원축소 시 데이터 손실있음 ( 최대한 유지하면서 축소시키는게 pca)