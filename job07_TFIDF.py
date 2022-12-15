#문장안에서 단어가 몇번나오는지의 빈도수를 파악하는 것
#문장 여러개 합치면 문서. 문서에서 빈도가 많으면 패널티를 줌
#문장끼리 유사도 찾기 > 전체 리뷰 데이터에서 '엘사'가 많이 나올경우 (모든 문장에서 나올 수 있으므로 유사도에서 제외시킴 =idf)가중치를 낮게 줌
#일부 문장에서 많이 나오면 곱해줌 tf (t= text f=frequency)


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.io import mmwrite, mmread #행렬을 읽고 저장하는
import pickle

df_reviews = pd.read_csv('./crawling_data/one_sentences.csv')
df_reviews.info()

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df_reviews['reviews'])
print(tfidf_matrix[0].shape)
with open('./models/tfidf.pickle','wb') as f: #모델 저장
    pickle.dump(tfidf, f)
mmwrite('./models/tfidf_movie_review.mtx', tfidf_matrix) #매트리스라서 mmwirte 사용 #결과값을 저장

#9만개의 단어들의 빈도수를 나타내는 행렬이 만들어짐
#99423의 단어 마다 빈도수의 역수를 곱함,
#빈도수가지고 좌표를 만들어서 점을 찍으면 빈도수가 같으면 좌표공간상에서 같은 위치에 있다
#문장의 유사도를 볼때는 방향을 봐야함,
#단어 유사도는 거리만 보면 되지만 , 문장의 빈도수를 볼때는 방향 거리 봐야함


