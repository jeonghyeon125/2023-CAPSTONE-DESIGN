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
    naver = train_naver.objects.filter(URL = vid.URL)
    naver_list = list(naver.body)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="outputs/training-2023-09-27_22-43-23")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--query_text_paths",
        type=str,
        nargs='+',
        # default=["query1.txt", "query2.txt", "query3.txt"],  # 여러 개의 query 파일을 지정
        default=naver_list,
    )
    parser.add_argument(
        "--total_text_path",
        type=str,
        # default="text.txt",
        default=stt.stt_result
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sentence_encoder = AutoModel.from_pretrained(args.model_name_or_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # with open(args.total_text_path, "r", encoding="utf-8") as f:
    #     all_sentences = f.readlines()
    all_sentences = args.total_text_path.split('.')
    
    # 문자열 앞뒤 공백 제거
    all_sentences = [txt.strip() for txt in all_sentences]

    sentence_embeddings = []  # 문장 임베딩 저장용
    for batch_sentence in tqdm(chunked(all_sentences, args.batch_size), total=len(all_sentences) // args.batch_size):
        sentence_embeddings.append(get_embedding(sentence_encoder, tokenizer, batch_sentence, device).detach().cpu().numpy())
    
    sentence_embeddings = np.concatenate(sentence_embeddings, axis=0)

    overall_avg_similarities = []

    for query_path in args.query_text_paths:
        with open(query_path, "r", encoding='utf-8') as query_file:
            input_text = query_file.read()

        print("-" * 100)
        print(f"Input:\n{input_text}\n")

        all_query = [input_text]  # 현재 query 파일 하나씩 처리
        avg_similarities = []

        for query in all_query:
            query_embedding = get_embedding(sentence_encoder, tokenizer, [query], device).detach().cpu().numpy()

            similarity = cosine_similarity(query_embedding, sentence_embeddings)[0]
            sentences_similarity = list(zip(all_sentences, similarity))
            sentences_similarity.sort(key=lambda x: x[1], reverse=True)

            logits = [o for _, o in sentences_similarity]
            avg_logits = sum(logits) / len(logits)

            print("-" * 100)
            print(f"Average probability: {avg_logits}\n")
            avg_similarities.append(avg_logits)

        overall_avg_similarity = sum(avg_similarities) / len(avg_similarities)
        overall_avg_similarities.append(overall_avg_similarity)

    final_avg_similarity = sum(overall_avg_similarities) / len(overall_avg_similarities)

    with open("output.txt", "w", encoding='utf-8') as f:
        for i, query_path in enumerate(args.query_text_paths):
            f.write(f"Query File {i + 1} - Overall Average Probability: {overall_avg_similarities[i]}\n")

        f.write(f"Final Overall Average Probability for All Queries: {final_avg_similarity}\n")

    print(f"Final Overall Average Probability for All Queries: {final_avg_similarity} \n")
