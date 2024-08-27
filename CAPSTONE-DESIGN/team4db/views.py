from django.shortcuts import render, redirect
from django.http import HttpResponse
def index(request):
    return HttpResponse("test")


# Create your views here.
import os
import requests
# 장고
from django.views.decorators.csrf import csrf_exempt
from mysite import settings
# DB 테이블
from .models import InOut, Video, STT, Keyword, train_naver, train_snu
from .models import adjective, adverb, dic_usr, eomi, exclamation, foreign, hanja, josa, noun, suffix, symbol, verb
# DB 쿼리
import sqlite3
# AWS
import boto3
from botocore.exceptions import NoCredentialsError
# 유튜브 영상 다운 
from pytube import YouTube
# 전처리
from konlpy.tag import Okt
from konlpy.tag import Hannanum
import pandas as pd
import numpy as np
import platform, re
# TF-IDF
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
# 크롤링
from bs4 import BeautifulSoup
import requests
import re
import datetime
from tqdm import tqdm
import sys
# 유사도 확인 실행
import torch
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
from more_itertools import chunked
from tqdm import tqdm
import numpy as np
# 모델 학습
from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import logging 
from datetime import datetime
import gzip
import csv
from easydict import EasyDict

@csrf_exempt
def main(request):
    if request.method == 'GET':
        return render(request, 'main.html')
    # elif request.method == 'POST':
    #     redirect('/s3db')


""" 사용자가 업로드한 파일을 다운로드 """
def register(request):
    if request.method == 'GET':
        return render(request, 'index.html')
    elif request.method == 'POST':
        DOWNLOAD_FOLDER = f'C:/Users/snu08/gradproject/mysite/' 

        # InOut 테이블에 url 저장
        inout = InOut()
        input_url = request.POST.get('input_url')
        inout.URL = input_url
        inout.save()

        # Video 테이블에 영상 저장
        video = Video()
        video.URL = InOut.objects.get(URL = input_url)
        yt = YouTube(inout.URL)
        video.vid = yt.streams.filter(only_audio=True).first().download()
        video.save()

        # STT 테이블에 url 저장
        stt = STT()
        check_null = STT.objects.filter(URL = input_url).exists()
        if(check_null == 0):
            stt.URL = InOut.objects.get(URL = input_url)
            stt.save()

        # Keyword 테이블에 url 저장
        keywords = Keyword()
        check_null = Keyword.objects.filter(URL = input_url).exists()
        if(check_null == 0):
            keywords.URL = InOut.objects.get(URL = input_url)
            keywords.save()

        # train_naver 테이블에 url 저장
        naver = train_naver()
        naver.URL = InOut.objects.get(URL = input_url)
        naver.save()

    return redirect('/s3db')


""" aws s3와 연동 및 stt 작업 수행 """

def upload_file_to_s3(request):
    # 로컬 파일을 s3로 업로드 후 로컬에서만 삭제
    s3 = boto3.client('s3', aws_access_key_id=settings.AWS_ACCESS_KEY_ID, aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY)
    last_row = Video.objects.last()
    file_name = str(last_row.vid)
    file_name = file_name[36:] # 나현
    # file_name = file_name[34:] # 정현
    bucket_name = settings.AWS_STORAGE_BUCKET_NAME
    key = f'media/{file_name}'
    s3.upload_file(file_name, bucket_name, key)

    file_path = f'C:/Users/User/Desktop/mysite/mysite/{file_name}'
    if os.path.exists(file_path):
        os.remove(file_path)

    # stt 작업 수행
    transcribe = boto3.client('transcribe')
    job_name = f'kjh_{str(last_row.id)}'
    # job_name = str(last_row.id)
    job_uri = f"s3://{settings.AWS_STORAGE_BUCKET_NAME}/media/{file_name}"
    transcribe.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={'MediaFileUri': job_uri},
        MediaFormat='mp4',
        LanguageCode='ko-KR',
        MediaSampleRateHertz=44100
    )
    while True:
        result = transcribe.get_transcription_job(TranscriptionJobName=job_name)
        if result['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
            break

    # STT 테이블에 stt 결과가 없다면 텍스트로 저장
    stt_row = STT.objects.get(URL = last_row.URL)
    if stt_row.stt_result is None:
        transcript_result = requests.get(result['TranscriptionJob']['Transcript']['TranscriptFileUri']).json()
        stt_row.stt_result = transcript_result['results']['transcripts'][0]['transcript']
        stt_row.save()

    return redirect('/keywords')
    # return render(request, 'main.html')


""" 텍스트 전처리 및 사용자 사전에 단어 추가 """
def preprocessings(request):

    """ 텍스트 전처리 """
    # 토큰화 함수
    def tokenize_by_space(text):
        tokens = text.split()
        return tokens
    
    # 품사 태깅 함수
    def pos_tagging(text):
        hannanum = Hannanum()  # hannanum 객체 초기화
        tagged_tokens = hannanum.pos(text)
        return tagged_tokens
    def okt_clean(tagged_text):
        cleaned_text = []
        for word, pos in tagged_text:
            if pos not in ['M', 'P', 'J', 'E', 'S', 'X']:  # 'J', 'E', 'S', 'X' 태그 제외
                cleaned_text.append((word, pos))
        return cleaned_text
    
    # 형식 전처리 함수
    def preprocessing(text):
        text = str(text)
        # 한, 영, 수 제외 모두 제거
        text = re.sub('[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9]',' ', text)
        # 중복 생성 공백값 제거
        text = re.sub('[\s]+', ' ', text)
        # 영문자를 소문자로 변환
        text = text.lower()
        return text
    
    # 불용어 제거
    def remove_stopwords(text):
        meaningful_words = []
        with open("C:/Users/User/DeskTop/mysite/mysite/dic/stopwords.txt", "r", encoding = 'utf-8') as f: # 나현
        # with open("C:/Users/snu08/gradproject/mysite/dic/stopwords.txt", "r", encoding = 'utf-8') as f: # 정현
            stops = f.read()
        stops = stops.split('\n')
        for i in range(len(text)):
            tokens = text[i].split(' ')
            for w in tokens:
                if not w in stops:
                    meaningful_words.append(w)
        return ' '.join(meaningful_words)
    
    # list 만드는 함수
    def load_word_list(tag):
        word_list = []
        for tokens in dataset["cleaned_text_data"]:
            for token in tokens:
                if token[1] == tag:
                    word_list.append(token[0])
        return word_list

    # 각각의 사전 불러오기
    noun_dict = noun.objects.values('word')
    verb_dict = verb.objects.values('word')
    adverb_dict = adverb.objects.values('word')
    adjective_dict = adjective.objects.values('word')
    josa_dict = josa.objects.values('word')
    eomi_dict = eomi.objects.values('word')
    symbol_dict = symbol.objects.values('word')
    foreign_dict = foreign.objects.values('word')
    hanja_dict = hanja.objects.values('word')

    # 데이터셋 로드
    stt_row = STT.objects.get(URL = Video.objects.last().URL)
    text = stt_row.stt_result
    dataset =  pd.DataFrame(text.split('.'))
    dataset.columns = ['TITLE']
    # 형식 전처리 적용
    dataset["preprocessing_text_data"] = dataset["TITLE"].apply(lambda x: preprocessing(x))
    # 토큰화
    dataset["tokenized_text_data"] = dataset["preprocessing_text_data"].apply(lambda x: tokenize_by_space(x))
    # 불용어 제거
    dataset["delete_text_data"] = dataset["tokenized_text_data"].apply(lambda x: remove_stopwords(x))
    # 품사 태깅
    dataset["tagging_text_data"] = dataset["delete_text_data"].apply(lambda x: pos_tagging(x))
    # 조사, 어미, 문장부호 제거
    dataset["cleaned_text_data"] = dataset["tagging_text_data"].apply(lambda x: okt_clean(x))

    # STT 테이블에 전처리 결과(preprocessing, cleaned) 저장
    stt_row.preprocessing = ''.join(dataset['preprocessing_text_data'])
    stt_row.cleaned = dataset['cleaned_text_data'].values.tolist()
    stt_row.save()

    """ 사용자 사전에 단어 추가 """
    # 태깅된 단어들
    noun_list = load_word_list('N')
    adverb_list = load_word_list('M')
    verb_list = load_word_list('P')
    josa_list = load_word_list('J')
    eomi_list = load_word_list('E')
    symbol_list = load_word_list('S')
    foreign_list = load_word_list('F')

    # 사용자 사전에 추가할 단어 찾고, 추가하는 함수
    def search_and_add(word_list, word_dict, word_type):
        # 사용자 사전에 추가할 단어 찾기
        words_to_add = set()
        if word_list is not None:
            for word in word_list:
                if word not in word_dict:
                    words_to_add.add(word)

        # 사용자 사전에 단어 추가
        dict = dic_usr()
        for word in words_to_add:
            dict.word = word
            dict.word_type = word_type
            dict.save()

    # 사용자 사전에 추가할 단어 찾고, 추가
    search_and_add(noun_list, noun_dict, 'Noun')
    search_and_add(verb_list, verb_dict, 'Verb')
    search_and_add(adverb_list, adverb_dict, 'Adverb')
    search_and_add(josa_list, josa_dict, 'Josa')
    search_and_add(eomi_list, eomi_dict, 'Eomi')
    search_and_add(symbol_list, symbol_dict, 'Symbol')
    search_and_add(foreign_list, foreign_dict, 'Foreign')

    """ TF-IDF 키워드 추출 """

    # 전처리된 용어를 행렬로 반환
    corpus = dataset["cleaned_text_data"].apply(lambda x: ' '.join(map(str,x)))
    corpus = remove_stopwords(corpus).split(' ')
    stt_row.cleaned = corpus
    stt_row.save()
    cvect = CountVectorizer(ngram_range=(1,1)) # ,max_df=0.9 ngram은 (1,1)이나 (1,2)가 적당한듯
    dtm = cvect.fit_transform(corpus)
    vocab = cvect.get_feature_names_out()
    df_dtm = pd.DataFrame(dtm.toarray(), columns = vocab)

    # 반환된 행렬에 TF-IDF 가중치 적용
    tfidf = TfidfTransformer()
    dtm_tfidf = tfidf.fit_transform(dtm)
    df_dtm_tfidf = pd.DataFrame(dtm_tfidf.toarray(), columns=vocab)

    # 가중치가 가장 높은 5개의 키워드 선별
    words = dtm_tfidf.max(axis=0).toarray().ravel()
    indexer = words.argsort()
    keywords = np.array(cvect.get_feature_names_out())
    keywords = keywords[indexer[-5:]]
    
    # Keywords 테이블에 키워드 저장
    keyword_row = Keyword.objects.get(URL = Video.objects.last().URL)
    try:
        keyword_row.keyword1 = keywords[0]
        keyword_row.keyword2 = keywords[1]
        keyword_row.keyword3 = keywords[2]
        keyword_row.keyword4 = keywords[3]
        keyword_row.keyword5 = keywords[4]
        keyword_row.save()
    except:
        keyword_row.keyword1 = ' '.join(keywords)
        keyword_row.save()

    return redirect('/crawling')


""" 네이버 크롤링 """
def crawling_naver(request):
    global keyword_dict

    if request.method == 'GET':
        keyword_row = Keyword.objects.get(URL = Video.objects.last().URL)
        # keyword_row = Keyword.objects.last()

        keyword_dict = {
            'keyword1': keyword_row.keyword1,
            'keyword2': keyword_row.keyword2,
            'keyword3': keyword_row.keyword3,
            'keyword4': keyword_row.keyword4,
            'keyword5': keyword_row.keyword5,
        }

        return render(request, 'index.html', {'keyword_dict': keyword_dict})
            
    elif request.method == 'POST':
        list_keywords = request.POST.get('search')

        # 접속 많이 할 경우 일어나는 에러 방지
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/98.0.4758.102"}

        # 크롤링할 url 생성하는 함수
        def crawUrl(search, startPg, endPg):
            urls = []
            for i in range(startPg, endPg + 1):
                page = PgNum(i)
                url = "https://search.naver.com/search.naver?where=news&sm=tab_pge&query=" + keyword + "&start=" + str(page)
                urls.append(url)
            return urls

        # url 형식에 맞는 페이지 만드는 함수
        def PgNum(num):
            if num == 1:
                return num
            elif num == 0:
                return num+1
            else:
                return num+9*(num-1)

        # html에서 원하는 속성 추출하는 함수
        def newsCrawler(articles,attrs):
            newscontent=[]
            for i in articles:
                newscontent.append(i.attrs[attrs])
            return newscontent

        # 크롤링
        def articles_crawler(url):
            originalHtml = requests.get(i,headers=headers)
            html = BeautifulSoup(originalHtml.text, "html.parser")

            naverUrl = html.select("div.group_news > ul.list_news > li div.news_area > div.news_info > div.info_group > a.info")
            url = newsCrawler(naverUrl,'href')
            return url

        # 키워드 입력
        keyword = list_keywords

        stpage = int(1) 
        edpage = int(1)

        url = crawUrl(keyword,stpage,edpage)

        #뉴스 크롤러 실행
        news_titles = []
        news_url =[]
        news_contents =[]
        news_dates = []

        for i in url:
            url = articles_crawler(url)
            news_url.append(url)


        #제목, 링크, 내용 1차원 리스트로 꺼내는 함수 생성
        def makeList(newlist, content):
            for i in content:
                for j in i:
                    newlist.append(j)
            return newlist

        news_url_1 = []
        makeList(news_url_1,news_url)

        #NAVER 뉴스만 남기기
        final_urls = []
        for i in tqdm(range(len(news_url_1))):
            if "news.naver.com" in news_url_1[i]:
                final_urls.append(news_url_1[i])
            else:
                pass

        for i in tqdm(final_urls):
            #각 기사 html get하기
            news = requests.get(i,headers=headers)
            news_html = BeautifulSoup(news.text,"html.parser")

            # 뉴스 제목 가져오기
            title = news_html.select_one("#ct > div.media_end_head.go_trans > div.media_end_head_title > h2")
            if title == None:
                title = news_html.select_one("#content > div.end_ct > div > h2")

            # 뉴스 본문 가져오기
            content = news_html.select("article#dic_area")
            if content == []:
                content = news_html.select("#articeBody")

            # 기사 텍스트만 가져오기
            content = ''.join(str(content))

            # html태그제거 및 텍스트 다듬기
            pattern1 = '<[^>]*>'
            title = re.sub(pattern=pattern1, repl='', string=str(title))
            content = re.sub(pattern=pattern1, repl='', string=content)
            pattern2 = """[\n\n\n\n\n// flash 오류를 우회하기 위한 함수 추가\nfunction _flash_removeCallback() {}"""
            content = content.replace(pattern2, '')

            news_titles.append(title)
            news_contents.append(content)

            try:
                html_date = news_html.select_one("div#ct> div.media_end_head.go_trans > div.media_end_head_info.nv_notrans > div.media_end_head_info_datestamp > div > span")
                news_date = html_date.attrs['data-date-time']
            except AttributeError:
                news_date = news_html.select_one("#content > div.end_ct > div > div.article_info > span > em")
                news_date = re.sub(pattern=pattern1,repl='',string=str(news_date))

            news_dates.append(news_date)

        # 데이터 프레임 만들기
        news_df = pd.DataFrame({'published_date':news_dates,'title':news_titles,'url_naver':final_urls,'body':news_contents})
        news_df = news_df.drop_duplicates(keep='first',ignore_index=True)

        # train_naver 테이블에 저장
        video_url = Video.objects.last().URL

        for index, row in news_df.iterrows():
            train_n = train_naver()
            train_n.URL = video_url
            train_n.url_naver = row['url_naver']
            train_n.title = row['title']
            train_n.published_date = row['published_date']
            train_n.body = row['body']
            train_n.save()

    # 세션에 키워드 저장
    request.session['crawling_keyword'] = keyword_dict
    
    return redirect('/similarity')
    # return render(request, 'index.html', {'video_url': video_url} )


""" 유사도 비교 """
def similarity(request):

    # 유사도 확인 실행(inference_v2.py)

    # 평균 풀링 함수
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]  # Transformer 모델 출력
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # 평균 풀링 수행 : 각 토큰의 임베딩 값 더함
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    # 문장 임베딩 추출 함수
    def get_embedding(model, tokenizer, sentences, device):
        with torch.no_grad():
            input_ids = tokenizer(
                sentences, 
                padding=True, 
                truncation=True, 
                return_tensors='pt'
            ).to(device)

            # 문장 임베딩 출력
            model_output = model(**input_ids)
            
            # 풀링된 문장 임베딩 계산
            features = mean_pooling(model_output, input_ids['attention_mask'])
            
        return features

    vid = Video.objects.last()
    stt = STT.objects.get(URL = vid.URL)
    naver = train_naver.objects.filter(URL = vid.URL)[:20]
    naver_list = []
    for i in naver:
        naver_list.append(i.body)

    args = EasyDict({
        'model_name_or_path':"outputs/training-2023-09-27_22-43-23", # 나현
        # 'model_name_or_path':"outputs/training-2023-10-29_13-09-39", # 정현
        'batch_size': 128,
        'query_text_paths': naver_list,
        'total_text_path': stt.stt_result
    })
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sentence_encoder = AutoModel.from_pretrained(args.model_name_or_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    all_sentences = args.total_text_path.split('.')
    
    # 문자열 앞뒤 공백 제거
    all_sentences = [txt.strip() for txt in all_sentences]
    sentence_embeddings = []  # 문장 임베딩 저장용
    for batch_sentence in tqdm(chunked(all_sentences, args.batch_size), total=len(all_sentences) // args.batch_size):
        sentence_embeddings.append(get_embedding(sentence_encoder, tokenizer, batch_sentence, device).detach().cpu().numpy())
    
    sentence_embeddings = np.concatenate(sentence_embeddings, axis=0)
    overall_avg_similarities = []
    for query_path in args.query_text_paths:
        input_text = str(query_path)
        all_query = [input_text]  # 현재 query 파일 하나씩 처리
        avg_similarities = []
        for query in all_query:
            query_embedding = get_embedding(sentence_encoder, tokenizer, [query], device).detach().cpu().numpy()
            similarity = cosine_similarity(query_embedding, sentence_embeddings)[0]
            sentences_similarity = list(zip(all_sentences, similarity))
            sentences_similarity.sort(key=lambda x: x[1], reverse=True)
            logits = [o for _, o in sentences_similarity]
            avg_logits = sum(logits) / len(logits)
            avg_similarities.append(avg_logits)
        overall_avg_similarity = sum(avg_similarities) / len(avg_similarities)
        overall_avg_similarities.append(overall_avg_similarity)
    final_avg_similarity = sum(overall_avg_similarities) / len(overall_avg_similarities)

    # 최종 유사도 저장
    # inout = InOut()
    inout_row = InOut.objects.get(URL = vid.URL_id)
    inout_row.similarity = float(final_avg_similarity)
    inout_row.save()

    similarity_data = inout_row.similarity
    # video_url = Video.objects.last().URL

    news_out = train_naver.objects.filter(URL = vid.URL_id).values('published_date', 'title', 'url_naver')[:11]
    news_df = pd.DataFrame(news_out)
    news_df = news_df.fillna('')
    # df_html = news_df.to_html(classes='table table-bordered', index=False, escape=False, border=0)

    # 0번 줄(첫 번째 행)을 제거한 DataFrame 생성
    news_df_without_first_row = news_df.iloc[1:]

    # Styler 객체 생성, 스타일 적용
    styler = news_df_without_first_row.style.set_table_attributes('class="table table-bordered"').set_properties(**{'border': '2px solid black', 'text-align': 'center'})

    # HTML로 변환
    df_html = styler.to_html(index=False)
    # df_html = news_df.to_html()

    keyword_dict = request.session.get('crawling_keyword', {})  # 세션에서 키워드 사전 가져오기
    
    return render(request, 'index.html', {'keyword_dict': keyword_dict,
            'news_url': df_html, 'similarity_data' : similarity_data} )