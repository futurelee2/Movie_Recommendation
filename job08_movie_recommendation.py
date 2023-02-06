import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from scipy.io import mmread
import pickle
from konlpy.tag import Okt
from gensim.models import Word2Vec
import re


def getRecommendation(cosin_sim):
    simScore = list(enumerate(cosin_sim[-1])) #인덱스를 알기위해서 enumerate사용 # sort를 하면 인덱스가 날라가서 인덱스의 타이틀을 찾아가기 힘듦
    #cosin_sim 형태: [[0.123  0.15433 ...]] = 코싸인값
    #cosin_sim  [[값]]의 형태로 1개의 값이 들어있어서 인덱싱을 해줘야함 (0으로 해도 상관없으나, 어떤 경우 앞에 어떤 값이 들어가게 되면 항상 마지막에 위치하기때문 -1사용)
    #print(simScore) # 형태: [(0,코싸인값),(1,코싸인값),...]
    simScore = sorted(simScore, key=lambda x:x[1], reverse=True) #정렬해서 코싸인 값이 큰 걸 파악, key값을 기준으로 reverse = 내림차순(큰값이 먼저 나오게)/ x = x[1]로 줌
    simScore = simScore[:11] #가장 유사도가 높은 11개 (제일 유사한게 자기 자신이 나옴, 겨울왕국>겨울왕국 추천, 맨앞에꺼 빼줘야함)
    print(simScore)
    movie_idx = [i[0] for i in simScore] #자기를 포함한 인덱스
    recMovieList = df_reviews.iloc[movie_idx, 0] #타이틀 리턴
    print(type(recMovieList))  #시리즈로 저장
    return recMovieList

df_reviews = pd.read_csv('./crawling_data/one_sentences.csv')
tfidf_matrix = mmread('./models/tfidf_movie_review.mtx').tocsr()
with open('./models/tfidf.pickle', 'rb') as f:
    tfidf = pickle.load(f)

#print(df_reviews)

#영화 제목 이용한 추천
movie_idx = df_reviews[df_reviews['titles'] =='겨울왕국 2 (Frozen 2)'].index[0] #인덱스값 = 숫자 가져오기
print(movie_idx)
cosin_sim = linear_kernel(tfidf_matrix[movie_idx], tfidf_matrix) #linear_kernel 코싸인값 찾아줌 (겨울왕국과 다른 값의 코싸인 값을 찾아줌)
recommendation = getRecommendation(cosin_sim)
#print(recommendation[1:11]) #자기자신 0번빼고 1부터~10까지
#print(cosin_sim)
#두벡터가 이루는 코싸인 값을 알아야 유사도 파악가능 > (코싸인이 1에 가까울수록 유사하다)
#90도는 서로 연관이 없고(코싸인 값 0) 180도는 상반되는말(코싸인값 -1)



#key word 이용한 추천
# embedding_model = Word2Vec.load('./models/word2vec_movie_review.model')
# key_word = '크리스마스'
# sim_word = embedding_model.wv.most_similar(key_word, topn=10)
# words = [key_word]
# for word, _ in sim_word: #단어 유사도
#     words.append(word)
# print(words)
# sentence = []
# count = 11
# for word in words:  #유사도가 높은 단어는 최대 count수 곱함, 마지막 값은 1번 곱해줌
#     sentence = sentence + [word]*count
#     count -= 1
# sentence = ' '.join(sentence)
# print(sentence)
# sentence_vec = tfidf.transform([sentence])
# consin_sim = linear_kernel(sentence_vec,tfidf_matrix)
# recommendation = getRecommendation(consin_sim)
# print(recommendation)

#문장으로 추천
# sentence = '화려한 액션과 소름 돋는 반전이 있는 영화'
# review = re.sub('[^가-힣]',' ',sentence)
# okt = Okt()
# token = okt.pos(review, stem=True)
# df_token = pd.DataFrame(token, columns=['word', 'class'])
# df_token = df_token[(df_token['class']=='Noun') |
#                     (df_token['class']=='Verb') |
#                     (df_token['class']=='Adjective')]
#
# words = []
# for word in df_token.word:
#     if 1 < len(word):
#         words.append(word)
# cleaned_sentence = ' '.join(words)
# print(cleaned_sentence)
# sentence_vec = tfidf.transform([cleaned_sentence])
# cosin_sim = linear_kernel(sentence_vec, tfidf_matrix)
# recommendation = getRecommendation(cosin_sim)
# print(recommendation)




