import pandas as pd
from konlpy.tag import Okt
import re

# KoNLPy이 에서는 여러 가지의 형태소 분석기를 제공
# 언어 처리를 하기 위해서는 문장을 일정 의미를 지닌 작은 단어들로 나누어야 합니다.
# 가장 기본이 되는 단어를 토큰(token)
# 말뭉치(혹은 문장)가 주어졌을 때, 이러한 토큰 단위로 나누는 작업을 토크나이징(tokenizing)

df = pd.read_csv('./crawling_data/review_final.csv')
df.info()
print(df.head())

df_stopwords = pd.read_csv('./stopwords.csv', index_col=0)
stopwords = list(df_stopwords['stopword'])
stopwords = stopwords +['안나','제니퍼','미국','중국','영화','감독','리뷰','연출',
                        '장면','주인공','되어다','출연','싶다','올해','엘사','개봉','아카리']

print(type(stopwords))

okt = Okt() #형태소 단위로 나누기
df['clean_reviews'] = None
count=0
for idx, review in enumerate(df.reviews):
# # enumerate :순서가 있는 자료형(list, set, tuple, dictionary, string)을 입력으로 받아 인덱스 값,요소 값 돌려줌
    count += 1 # 점찍기
    if count % 10 ==0:
        print('.', end='')
    if count % 1000 == 0:
        print()

    review = re.sub('[^가-힣]', ' ', review) # ^글자 빼고, 빈칸으로 만들기, abstract 변수 안에있는 값을
    df.loc[idx, 'clean_reviews'] = review
    token = okt.pos(review, stem=True)  # pose: 형태소와 품사묶어서 튜플로 만들어주는 함수, stem은 원형으로 바꿔줌 > 결국 명사, 동사만 남음
    df_token = pd.DataFrame(token, columns=['word','class']) #word형태소, class품사
    df_token = df_token[(df_token['class']=='Noun') |
                        (df_token['class'] == 'Verb') |
                        (df_token['class'] == 'Adjective')] #명사, 동사, 형용사만 추출
    #print(df_token)

    words = []
    for word in df_token.word:
        if len(word) > 1:
            if word not in list(df_stopwords.stopword):
                words.append(word)
    cleaned_sentence = ' '.join(words)
    #print(cleaned_sentence)
    #print(type(cleaned_sentence))
    df.loc[idx,'clean_reviews'] = cleaned_sentence
print(df.head(30))
df.dropna(inplace=True)
df.to_csv('./crawling_data/cleaned_reviews_2016_2022.cvs', index=False)


# print(df_token.head(30))
# print(df.clean_reviews)
# print(token)
