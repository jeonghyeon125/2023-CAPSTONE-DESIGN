# 모델 학습
from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import logging 
from datetime import datetime
import gzip
import csv

# 디버그 정보 표준 출력
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

# 데이터셋
sts_dataset_path = 'C:/Users/User/Desktop/mysite/mysite/dataset.csv.gz'

# 데이터셋 읽어오기
# RoBERTa 한국어 언어 모델 사용
model_name = 'jhgan/ko-sroberta-multitask' # 'klue/bert-base' or 'jhgan/ko-sroberta-multitask'
train_batch_size = 32
num_epochs = 5
model_save_path = 'outputs/training-' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# SentenceTransformer 모델 로드
model = SentenceTransformer(model_name)

# 데이터셋 읽어와서 DataLoader로 변환
logging.info("Read STSbenchmark train dataset")

train_samples = []
dev_samples = []
test_samples = []
with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter=',', quoting=csv.QUOTE_NONE)
    for row in reader:
        score = float(row['score']) / 5.0  # score 정규화 (0~1 사이)
        inp_example = InputExample(texts=[row['sentence1'], row['sentence2']], label=score)

        if row['split'] == 'dev':               # 검증
            dev_samples.append(inp_example)
        elif row['split'] == 'test':            # 테스트
            test_samples.append(inp_example)
        else:                                   # 훈련
            train_samples.append(inp_example)


# 데이터 DataLoader 변환
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
train_loss = losses.CosineSimilarityLoss(model=model)   # 손실 함수, 코사인 유사도 기반으로 손실 계산


logging.info("Read STSbenchmark dev dataset")
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')


# 모델 훈련 설정
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))


# 모델 학습
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=model_save_path)


##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################

# 모델 평가
model = SentenceTransformer(model_save_path)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-test')
test_evaluator(model, output_path=model_save_path)