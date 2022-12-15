import sys
from PyQt5. QtWidgets import * # 파일 내 모든 파일 가져오기 > *
from PyQt5 import uic
from PyQt5.QtCore import QStringListModel
import pandas as pd
from scipy.io import mmread
import pickle
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import linear_kernel
import re
from konlpy.tag import Okt


form_window = uic.loadUiType('./movie_recommendation.ui')[0] #form window 변수이름/ = 윈도우uic를 클래스로 만들어주는 것

class Exam(QWidget, form_window): #다중 상속 (클래스를 여러개 넣어주기 가능) / qwidget: 위젯으로 만들기 위한 기본적인 것/ form window 디자인 클래스
    def __init__(self): #class에서 선언하는 첫번째 매개변수는 무조건 self > 클래스 만들려면 무조건!!  def __init__(self): 있어야함
        super().__init__()
        self.setupUi(self) #라벨. 푸쉬버튼 #내가 만든 ui를 여기서 세팅할꺼야

        self.tfidf_matrix = mmread('./models/tfidf_movie_review.mtx').tocsr()
        with open('./models/tfidf.pickle', 'rb') as f:
            self.tfidf = pickle.load(f)
        self.embedding_model = Word2Vec.load('./models/word2vec_movie_review.model')

        self.df_reviews = pd.read_csv('./crawling_data/one_sentences.csv')
        self.titles = list(self.df_reviews['titles'])
        self.titles.sort()
        for title in self.titles:
            self.combo_box.addItem(title)

        model = QStringListModel() # 자동완성 모듈을 위한 리스트를 만들어줌
        model.setStringList(self.titles) #위에 만든 타이틀을 리스트에 넣어줌
        completer = QCompleter() #자동완성 프로그램
        completer.setModel(model)
        self.line_edit.setCompleter(completer) #입력창에 자동완성을 적용시켜줌

        self.combo_box.currentIndexChanged.connect(self.combobox_slot) #사용자가 어떤걸 누르면 인덱스가 바뀜
        self.btn_recommend.clicked.connect(self.btn_slot)


    def recommendation_by_movie_title(self,title):
        movie_idx = self.df_reviews[self.df_reviews['titles'] == title].index[0]
        cosin_sim = linear_kernel(self.tfidf_matrix[movie_idx], self.tfidf_matrix)  # linear_kernel 코싸인값 찾아줌
        recommendation = self.getRecommendation(cosin_sim)
        recommendation = '\n'.join(list(recommendation[1:]))
        self.lbl_recommend.setText(recommendation)

    def recommendation_by_key_word(self, key_word):
        sim_word = self.embedding_model.wv.most_similar(key_word, topn=10)
        words = [key_word]
        for word, _ in sim_word:  # 단어 유사도
            words.append(word)
        print(words)
        sentence = []
        count = 11
        for word in words:  # 유사도가 높은 단어는 최대 count수 곱함, 마지막 값은 1번 곱해줌
            sentence = sentence + [word] * count
            count -= 1
        sentence = ' '.join(sentence)
        print(sentence)
        sentence_vec = self.tfidf.transform([sentence])
        consin_sim = linear_kernel(sentence_vec, self.tfidf_matrix)
        recommendation = self.getRecommendation(consin_sim)
        recommendation = '\n'.join(list(recommendation[1:])) #\n 줄바꿈
        self.lbl_recommend.setText(recommendation)

    def recommendation_by_sentence(self, key_word):
        sentence = key_word
        review = re.sub('[^가-힣]',' ',key_word)
        okt = Okt()
        token = okt.pos(review, stem=True)
        df_token = pd.DataFrame(token, columns=['word', 'class'])
        df_token = df_token[(df_token['class']=='Noun') |
                            (df_token['class']=='Verb') |
                            (df_token['class']=='Adjective')]

        words = []
        for word in df_token.word:
            if 1 < len(word):
                words.append(word)
        cleaned_sentence = ' '.join(words)
        print(cleaned_sentence)
        sentence_vec = self.tfidf.transform([cleaned_sentence])
        cosin_sim = linear_kernel(sentence_vec, self.tfidf_matrix)
        recommendation = self.getRecommendation(cosin_sim)
        recommendation = '\n'.join(list(recommendation[1:]))
        self.lbl_recommend.setText(recommendation)


    def btn_slot(self):
        key_word = self.line_edit.text() #검색창 글자를 문자로 받기
        if key_word in self.titles: #타이틀에 완벽하게 맞으면
            self.recommendation_by_movie_title(key_word)

        elif key_word in list(self.embedding_model.wv.index_to_key): # 리뷰에 있는 키워드라면
            self.recommendation_by_key_word(key_word)
        else:
            self.recommendation_by_sentence(key_word)



    def combobox_slot(self):
        title = self.combo_box.currentText()
        self.recommendation_by_movie_title(title)

    def getRecommendation(self,cosin_sim):
        simScore = list(enumerate(cosin_sim[-1]))  # 인덱스를 알기위해서 enumerate사용 # sort를 하면 인덱스가 날라가서 인덱스의 타이틀을 찾아가기 힘듦
        # cosin_sim  [[값]]의 형태로 1개의 값이 들어있어서 인덱싱을 해줘야함 (0으로 해도 상관없으나, 어떤 경우 앞에 어떤 값이 들어가게 되면 항상 마지막에 위치하기때문 -1사용)
        simScore = sorted(simScore, key=lambda x: x[1], reverse=True)  # 정렬해서 코싸인 값이 큰 걸 파악, reverse = 내림차순(큰값이 먼저 나오게)
        simScore = simScore[:11]  # 가장 유사도가 높은 11개 (제일 유사한게 자기 자신이 나옴, 겨울왕국>겨울왕국 추천, 맨앞에꺼 빼줘야함)
        movie_idx = [i[0] for i in simScore]  # 자기를 포함한 인덱스
        recMovieList = self.df_reviews.iloc[movie_idx, 0]  # 타이틀 리턴
        return recMovieList


    # object name 봐야함/ #clicked시그널 발생햇을때 text는 속성값 변경

if __name__ == "__main__": #모듈로 써먹기 위해 습관적으로 하는 것(기본적인것)
    app = QApplication(sys.argv)  #함수 혹은 생성자이나 여기서 생성자 (QApplication: 윈도우를 만들고 어플을 어플로 동작하게 만드는 기능 가짐)
    mainWindow = Exam() #실제로 객체를 만들고
    mainWindow.show() #화면에 출력하기
    sys.exit(app.exec_()) #어플리케이션: 사용자가 액션을 취하는 것을 (입력에대한) 처리해줌 / exit: 파이썬 소스코드 종료 / 윈도우가 종료되면 exit 실행되면서 종료


#슬롯에 연결하면 연결된 슬롯이 동작함
#시그널은 버튼을 누리면 발생>