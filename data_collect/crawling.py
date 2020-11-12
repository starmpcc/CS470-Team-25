from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from bs4 import BeautifulSoup
import requests

import datetime as dt
import os
import zipfile   
import shutil
import time


url_list = []
f = open("url.txt", "r")
for x in f:
    url_list.append(x)
f.close()


options = webdriver.ChromeOptions()
mobile_emulation = {"deviceName": "Nexus 5"}
options.add_experimental_option("mobileEmulation", mobile_emulation)

options.add_argument('headless')
options.add_argument('window-size=1920x1080')
options.add_argument("disable-gpu")

cat_num = 89

for url in url_list:

    driver = webdriver.Chrome('./chromedriver', chrome_options=options)
    # 모든 동작마다 크롬브라우저가 준비될 때 까지 최대 5초씩 대기
    #driver.implicitly_wait(5)

    driver.get(url)
    time.sleep(3)

    img_list = []

    for i in range(0, 6):
        # 현재 브라우저에 표시되고 있는 소스코드 가져오기
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        #import pdb; pdb.set_trace()
        # srcset이라는 속성을 포함하는 모든 이미지 태그 가져오기 --> 리스트형으로 반환됨
        img = soup.select("img[srcset]")
        
        # 미리 준비한 리스트에 결합시킴
        img_list += img
        
        # 동일한 항목에 대한 중복제거
        img_list = list(set(img_list))
        
        # 수집 과정을 출력한다.
        print("%04d번째 페이지에서 %02d건 수집함 >> 누적 데이터수: %05d" % (i+1, len(img), len(img_list)))
        
        # 스크롤을 맨 아래로 이동
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        # 다음 컨텐츠가 로딩되는 동안 1초씩 대기
        time.sleep(1)


    src_list = []
    for t in img_list:
        srcset = t.attrs['srcset']              # srcset 속성 가져오기
        srcset_list = srcset.split(",")         # 쉼표 단위로 추출
        item = srcset_list[len(srcset_list)-1]  # 이미지 해상도가 가장 큰 마지막 원소를 선택
        url = item[:item.find(" ")]             # 첫 번째 글자부터 마지막 공백문자 전까지 잘라냄
        src_list.append(url)                    # 준비한 리스트에 추출결과 넣기
        
    # 중복제거를 위해 집합으로 변경 후 리스트로 다시 변환
    src_list = list(set(src_list))

    #datetime = dt.datetime.now().strftime("%y%m%d_%H%M")
    #cat_num = 1
    dirname = "cat/insta_%s" % (cat_num)

    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.120 Safari/537.36"
    session = requests.Session()
    session.headers.update( {'User-agent': user_agent, 'referer': None} )

    # 저장된 이미지 수를 카운트하기 위한 변수
    count = 0

    # 폴더 생성하기
    if not os.path.exists(dirname):
        os.mkdir(dirname)
        
    # 폴더 이름으로 빈 압축파일 생성
    #zipfile_name = '%s.zip' % dirname
    #insta_zip = zipfile.ZipFile(zipfile_name, 'w')

    # 이미지 URL수 만큼 반복
    for image_url in src_list:
        # 카운트 증가
        count += 1

        # 파일이 저장될 경로 생성
        path = "%s/%04d.jpg" %(dirname, count)

        # 예외처리를 동반한 이미지 다운로드
        try:
            # 이미지 주소를 다운로드를 위해 stream 모드로 가져온다.
            r = session.get(image_url, stream=True)

            # HTTP 상태코드가 성공을 의미하는 값이 아니라면 에러로 간주하고 except 블록으로 강제 이동
            if r.status_code != 200:
                raise Exception
                
            # 추출한 데이터를 바이너리(이진값) 쓰기 모드로 저장 -> 'wb'
            with open(path, 'wb') as f:
                f.write(r.raw.read())
                #print( "[Ok] %s(이)가 저장되었습니다." % path )
            
            # 압축파일에 다운로드 된 이미지 추가
            #insta_zip.write(path)
            
        except:
            # 이미지 다운로드 실패시 다음 이미지를 시도하기 위해 반복의 조건식으로 이동함
            #print( "[Error] %s(은)는 저장에 실패했습니다." % path )
            continue
    print( " %d개 저장되었습니다." % count )

    cat_num += 1
