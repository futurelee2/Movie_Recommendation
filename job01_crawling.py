from selenium import webdriver
import pandas as pd
from selenium.common.exceptions import NoSuchElementException
import time

options = webdriver.ChromeOptions()

options.add_argument('lang=ko_KR')
driver = webdriver.Chrome('./chromedriver.exe', options=options)

#selenium은 웹사이트 테스트를 위한 도구로 브라우저 동작을 자동화 가능.
# 프로그래밍으로 브라우저 동작을 제어해서 마치 사람이 이용하는 것 같이 웹페이지를 요청하고 응답을 받아올 수 있음.


#지정되어있어 for문 안에서 불필요한 것들 빼버림
review_button_xpath = '//*[@id="movieEndTabMenu"]/li[6]/a' #동영상 없으면 li[5], 있으면 li[6]
review_num_path = '//*[@id="reviewTab"]/div/div/div[2]/span/em'
review_xpath = '//*[@id="content"]/div[1]/div[4]/div[1]/div[4]'

# 컬럼명은 [ 'titles', 'reviews'] 로 통일
# 파일명은 'reviews_{}.csv'. format(연도)
# crawling 코드 완성되면 PR 부탁드립니다 ^,^

your_year = 2016
for page in range(30,60): #영화 페이지
    url = 'https://movie.naver.com/movie/sdb/browsing/bmovie.naver?open={}&page={}'.format(your_year,page)
    titles = [] #페이지 20개 긁을때마다 저장
    reviews = []

    try:
        for title_num in range(1,21): # 영화 제목 수 20개
            driver.get(url)
            time.sleep(0.1)
            movie_title_xpath = '//*[@id="old_content"]/ul/li[{}]/a'.format(title_num)
            title = driver.find_element('xpath', movie_title_xpath).text
            print('title',title)

            driver.find_element('xpath', movie_title_xpath).click()
            time.sleep(0.1)

            try:
                driver.find_element('xpath', review_button_xpath).click()
                time.sleep(0.1)


                #리뷰 건수
                review_num = driver.find_element('xpath', review_num_path).text
                review_num = review_num.replace(',','') #리뷰개수 1000개 넘어가면 콤마찍힘 > 콤마없애기
                review_range = (int(review_num)-1) // 10+1  #10으로 나눠서 떨어졌을 경우 때문, 몫 반환 (나머지 반환 %)
                #문자열임으로 int로 바꿔줘야함, 한페이지당 10개 리뷰

                if review_range>3: #30페이지까지 리뷰 들고오기
                    review_range=3

                for review_page in range(1, review_range+1):
                    review_page_button_xpath= '//*[@id="pagerTagAnchor{}"]/span'.format(review_page)
                    driver.find_element('xpath',review_page_button_xpath).click()
                    time.sleep(0.1)

                    for review_title_num in range(1,11):     #리뷰에서 한 페이지당 리뷰수 10개
                        review_title_xpath = '//*[@id="reviewTab"]/div/div/ul/li[{}]/a'.format(review_title_num)
                        driver.find_element('xpath', review_title_xpath).click()
                        time.sleep(0.1) #로딩할 시간 필요

                        try:
                            review = driver.find_element('xpath',review_xpath).text
                            titles.append(title)
                            reviews.append(review)
                            driver.back() #리뷰 읽고 다른 리뷰를 읽기위해 다시 돌아감

                        except:
                            print('review',page,title_num,review_title_num)
                            driver.back() #리뷰 읽기위해 클릭했을 때, 리뷰페이지 안에서 에러가 났을 경우 다시 다른 리뷰를 읽기위해 back 필요

            except:
                print('review button', page,title_num)

        df = pd.DataFrame({'titles':titles, 'reviews':reviews})
        df.to_csv('./crawling_data/reviews_{}_{}page.csv'.format(your_year,page),index=False)
    except:
        print('error', page,title_num)






