# 유사도 확인 실행
import torch
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
from more_itertools import chunked
from tqdm import tqdm
import numpy as np

from .models import InOut, Video, STT, train_snu, train_naver

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


if __name__ == "__main__":
    vid = Video.objects.last()
    stt = STT.objects.get(URL = vid.URL)
    naver = train_naver.objects.get(URL = vid.URL)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="outputs/training-2023-09-27_22-43-23")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--query_text_path",
        type=str,
        # default="query.txt",    # 비교 파일 넣기
        default = naver.body
    )
    parser.add_argument(
        "--total_text_path",
        type=str,
        # default="text.txt",     # 비교가 필요한 파일 넣기
        defualt = stt.stt_result
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sentence_encoder = AutoModel.from_pretrained(args.model_name_or_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # with open(args.query_text_path, "r", encoding="utf-8") as f:
    #     all_query = f.readlines()
    all_query = stt.stt_result.split('.')
    
    # with open(args.total_text_path, "r", encoding="utf-8") as f:
    #     all_sentences = f.readlines()
    all_sentences = naver.body.split('.')

    # 문자열 앞뒤 공백 제거
    all_sentences = [txt.strip() for txt in all_sentences]

    sentence_embeddings = []    # 문장 임베딩 저장용
    for batch_sentence in tqdm(chunked(all_sentences, args.batch_size), total=len(all_sentences) // args.batch_size):
        sentence_embeddings.append(get_embedding(sentence_encoder, tokenizer, batch_sentence, device).detach().cpu().numpy())
    
    sentence_embeddings = np.concatenate(sentence_embeddings, axis=0)
    
    # with open("output.txt", "w", encoding = 'utf-8') as f:
    #     # query.txt 내용 읽고 텍스트로 저장
    #     with open("query.txt", "r", encoding = 'utf-8') as query_file:
    #         input_text = query_file.read()
            
        # print("-" * 100)
        # print(f"Input:\n{input_text}\n")
        # print("-" * 100)

        # f.write(f"Input:\n{input_text}\n\n")

    for query in all_query:
        query_embedding = get_embedding(sentence_encoder, tokenizer, [query], device).detach().cpu().numpy()
        
        similarity = cosine_similarity(query_embedding, sentence_embeddings)[0]
        sentences_similarity = list(zip(all_sentences, similarity))
        sentences_similarity.sort(key=lambda x: x[1], reverse=True)
        
        logits = [o for _, o in sentences_similarity]
        avg_logits = sum(logits) / len(logits)
        
        # print("-" * 100)
        # print(f"Average probability: {avg_logits}\n")
        # f.write(f"Average probability: {avg_logits}\n")

    all_logits = [o for _, o in sentences_similarity]
    overall_avg_similarity = sum(all_logits) / len(all_logits)
    
    # 총 유사도 결과
    inout = InOut.objects.get(URL = vid.URL)
    inout.similarity = overall_avg_similarity
    # print(f"Overall Average Probability for All Queries: {overall_avg_similarity} \n")
    # f.write(f"Overall Average Probability for All Queries: {overall_avg_similarity}\n")