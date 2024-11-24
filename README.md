![썸네일](https://github.com/user-attachments/assets/da3688c5-5348-4ed5-afc2-1f01fd4e646c)

# 2023-CAPSTON-DESIGN
> Identification of Internet news reliability using TF-IDF and RoBERTa models <br>
> TF-IDF와 RoBERTa 모델을 이용한 인터넷 뉴스 신뢰도 판별 <br>

> 한국정보처리학회 ASK 2023 춘계 학술대회 논문 게재 [학술대회 논문 바로가기](https://www.riss.kr/search/detail/DetailView.do?p_mat_type=1a0202e37d52c72d&control_no=ee80b5202c9e15e747de9c1710b0298d&keyword=) <br>
> 홍익대학교 제 20회 소프트웨어융합학과 학술제 우수상 

 
<br>

## ⚙Tech Stack
<p><strong> Language <br></strong>
 <img src="https://img.shields.io/badge/Java-007396?style=for-the-badge&logo=java&logoColor=white"/>
 <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white"/> 
 
<p><strong> Cloud storage <br></strong>
<img src="https://img.shields.io/badge/AWS S3-569A31?style=for-the-badge&logo=Amazon S3&logoColor=white">

</p>
<p><strong> Backend / DataBase <br></strong>
<img src="https://img.shields.io/badge/Django-092E20?style=for-the-badge&logo=Django&logoColor=white"/>
</p>
<p><strong> Frontend <br></strong>
 <img alt="HTML" src="https://img.shields.io/badge/html5-E34F26?style=for-the-badge&logo=html5&logoColor=white"> 
<img alt="CSS" src="https://img.shields.io/badge/css-1572B6?style=for-the-badge&logo=css3&logoColor=white"> 
<img alt="JavaScript" src="https://img.shields.io/badge/javascript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black"> 
</p>
<p><strong> DevTools <br></strong>
<img src="https://img.shields.io/badge/VSCode-007ACC?style=for-the-badge&logo=VisualStudioCode&logoColor=white"/>

<p><strong> VCS <br></strong>
<img src="https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=Git&logoColor=white"/> 
<img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=GitHub&logoColor=white"/>
  
<br><br>

## 🗂️ Project Planning

### 1. Project Outline
<br>

![image](https://github.com/user-attachments/assets/e04ab22b-3d8c-45c9-a138-568da7d104bd)

<br>

한국언론진흥재단의 소셜미디어 이용자 조사에 따르면 허위 정보 접촉 여부가 77.2%에 달했으며, <br>
YouTube가 허위 정보가 주로 확산되는 미디어 1순위로 꼽혔다.

또한, 디지털 환경의 발달로 누구나 제약 없이 콘텐츠를 제작할 수 있게 되면서 가짜뉴스로 인한 피해 사례가 증가하고 있다. <br>
이에 우리는 Hongik TruthGuard 라는 진위판단에 도움을 주는 시스템을 개발하게 되었다. <br>

이 시스템은 사용자가 진위 판단이 필요한 영상의 URL을 입력하면 영상에서 키워드를 추출하고, <br>
이를 기반으로 뉴스를 검색해 유사도를 제공한다. 이를 통해 사용자가 인터넷 뉴스 신뢰도 판별에 올바른 판단을 할 수 있도록 돕는다. <br>
이 프로젝트의 구체적인 목표는 다음과 같다.

<h3>- Quantitative Goal 정량적 목표</h3>
1. 키워드 기반 웹 크롤링 시 사용자들에게 70% 이상의 관련된 기사 제공 <br>
2. 데이터 전처리 과정에서 나타나는 새로운 단어를 사용자 사전에 추가하여 학습함으로써 전처리 과정에서의 텍스트 처리 정확도 향상

<br>

<h3>- Qualitative Goal 정성적 목표</h3>
1. 동영상 데이터의 텍스트 데이터 추출을 통한 판별 범위 확장 <br>
2. 사용자들의 올바른 정보 습득 <br>
3. 뉴스 제작자들의 윤리적인 책임 의식 강화 기여

<br>

### 2. System Architecture
<br>

![architecture-removebg-preview](https://github.com/user-attachments/assets/f072548b-ccef-4f15-ad3c-9056f3d444a0)

<br>

## 🛠️ Develop

### Process

### 1. URL 입력 <br><br>

### 2. STT & DB 저장 : YouTube의 영상 데이터를 통한 음성 데이터 추출 적용 <br>
 a. 동영상 링크를 받아와 진행하여 사용자가 원하는 동영상의 내용의 데이터 사용 <br>
 b. 판별 범위가 기존의 인터넷 기사에서 온라인 동영상 플랫폼 확장 <br><br>
 
### 3. 텍스트 전처리 : 텍스트 데이터들의 전처리 및 새로운 단어 사용자 사전 추가 기능 적용 <br>
![image](https://github.com/user-attachments/assets/fd735c3f-7d9a-459c-9a4e-e79ba95d3c71) 
<br>
 a. Konlpy 에서 제공하는 라이브러리 사용, Okt - 조사, 어미 제거, Hannanum - 품사 태깅 <br>
 b. 불용어 제거 <br>
 c. 사전에 등록되지 않은 단어 (신조어, 고유명사 등) 발견 시 사용자 사전에 새로 추가 <br><br>
 
### 4. TF-IDF 및 웹 크롤링 : 전처리 된 데이터들을 통한 키워드 추출 및 키워드 기반 웹 크롤링 <br>
![image](https://github.com/user-attachments/assets/54cfbe7f-4bb8-4b6a-8dfb-378f6a4955d4)
![image](https://github.com/user-attachments/assets/947ccbb1-fa0d-43d6-a672-eed94a3681d5)

<br>
 a. 전처리 된 용어를 행렬로 반환 <br>
 b. 가중치가 가장 높은 5개의 키워드 선별 <br>
 c. 선별된 키워드 기반 네이버 뉴스 기사 크롤링 및 관련 기사 링크 제공 <br><br>
 
### 5. 유사도 계산 및 제공 <br>
![image](https://github.com/user-attachments/assets/11e67428-7971-4980-a238-bb941b97f707)

<br>

## 💾 Project Implementation

https://github.com/user-attachments/assets/1c3ff290-3e8d-4d8a-af64-bd9452fc76e3

<br>

## 📖 Performance Analysis
| Feature | SNU FactCheck (국내) | PolitiFact (국외) | Snopes (국외) | Hongik TruthGuard | Originality and Differentiation of Our Technology |
|---------|---------------|------------|--------|-------------------|--------------------------------------------------|
| 택스트 내용 분석 | O | O | O | X | 동영상의 내용을 텍스트로 전환하여 다른 기사와의 유사도를 비교한다. |
| 동영상 내용 분석 | X | X | X | O | 다른 사이트들은 오직 텍스트 기사만 분석한다. |
| 다양한 기사 제공 | O | O | O | O | TF-IDF 키워드 기반 검색
| 사용자가 분석이 필요한 내용 선택 | X | X | X | O | 단순 기사들의 나열 형태가 아닌 사용자가 직접 진위판별을 원하는 YouTube 영상의 URL을 입력한다. |

<br>

## ✨ Summary & Expected effect

### 1. 분석 범위 확장
텍스트 위주였던 분석 대상을 동영상으로 확장
프로젝트를 진행하기 위하여 YouTube 플랫폼으로 한정을 두었으나, 이는 다른 동영상 플랫폼으로 확장이 가능하며, 더 나아가 챗봇 형태의 구현까지도 기대할 수 있음

### 2. 올바른 정보 제공
사용자들의 올바른 정보 습득 및 뉴스 제작자들의 윤리 책임 의식 강화

<br>

## 👨‍👩‍👧‍👦 Member

| [김나현](https://github.com/NAHYEON0713) | [김정현](https://github.com/jeonghyeon125) | [손채영](https://github.com/caheyoun9) |
|:--------------------------------------:|:----------------------------------------:|:-------------------------------------:|
| Leader, Developer <br> 데이터 전처리, RoBERTa 모델 학습 | Developer <br> 데이터 전처리, TF-IDF | Developer <br> STT 추출, 웹 크롤링 및 웹 개발 |

<br>

## 📚 Memoir

| &nbsp;&nbsp;팀&nbsp;원&nbsp;&nbsp;&nbsp; | 회고록 |
| :--------------------------------------: | ------ |
|                  김나현                  | &nbsp; |
|                  김정현                  | &nbsp; |
|                  손채영                  | &nbsp; |

<br>

## 프로젝트 요약
![2023_capstone-design_summary](https://github.com/user-attachments/assets/17d9307d-e0d8-4ebf-a373-0530da8da17e)
<br>

| 발표자료 |
| ------ |
|https://drive.google.com/file/d/19fwdIyjsGSP5xRfnqJFEjb1UhbyqqCH6/view?usp=sharing|

<br>
