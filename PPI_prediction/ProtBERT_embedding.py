from transformers import BertTokenizer, BertModel
import torch
import numpy as np

"""
Make ProtBERT embeddings for protein sequences
input: text file with uniprot id and sequence
output: npz file with uniprot id and ProtBERT embedding
"""

# ProtBERT 모델과 토크나이저 로드

tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
# HuggingFace의 사전학습된 ProtBERT 모델에서 사용하는 **토크나이저(tokenizer)**를 로드
# 이 tokenizer는 아미노산 서열을 공백으로 구분된 토큰으로 분리해 모델에 전달 가능한 형태로 바꿔줌
# do_lower_case=False
# ProtBERT는 **대문자 아미노산 코드 (A, M, Q 등)**를 기반으로 훈련되었기 때문에, 
# 입력을 소문자로 변환하면 안 됨 → 그래서 False로 설정

model = BertModel.from_pretrained("Rostlab/prot_bert")
# ProtBERT의 **사전 학습된 BERT 모델 파라미터(weight)**를 불러옴
# 이 모델은 HuggingFace에서 관리하는 "Rostlab/prot_bert" 이름으로 업로드된 모델
# BertModel은 텍스트(여기서는 단백질 서열)를 받아 각 토큰과 전체 시퀀스에 대한 벡터 임베딩을 생성
# 출력은:
# last_hidden_state: [batch_size, seq_len, hidden_dim=1024]
# pooler_output (선택적)


# GPU 사용 가능하면 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)   # 모델을 위에서 지정한 device로 옮김
model.eval()  # 평가 모드
# 추론(inference) 시에는 반드시 eval()을 호출해야 출력이 일관되고 정확함


def _preprocess_sequence(seq):   # 공백을 아미노산 사이에 삽입
    return " ".join(list(seq))
# ProtBERT 모델은 공백으로 구분된 아미노산 토큰을 입력으로 받기 때문에, 각 아미노산 사이에 공백을 삽입

@torch.no_grad()   # 그래디언트 계산을 비활성화하여 메모리 사용량↓, 속도↑
def _embed_sequence(sequence):
    sequence = _preprocess_sequence(sequence)   # 데이터 전처리
    inputs = tokenizer(sequence, return_tensors="pt")   # 전처리한 문자열(Sequence)를 PyTorch 텐서로 변환(return_tensor="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}   # 위에서 만든 토큰 텐서들을 GPU 또는 CPU로 이동

    outputs = model(**inputs)   # 모델에 입력을 넣고 추론 실행
    # 주요 속성 - outputs.last_hidden_state: [batch_size, seq_len, hidden_dim] → 전체 시퀀스 각 토큰의 임베딩
    embedding = outputs.last_hidden_state[:, 0, :]   # ProtBERT는 BERT 구조이므로 첫 번째 토큰에 시퀀스 전체 정보를 요약한 벡터가 있음
    return embedding.cpu()   # GPU에서 계산된 결과를 CPU 메모리로 이동, 추후 저장 (np.save 등) 또는 후처리를 위해


@torch.no_grad()
def _sliding_embed(sequence, window_size=1024, stride=512):
    sequence = sequence.replace(" ", "")  # 혹시 모를 공백 제거
    embeddings = []

    # 슬라이딩 윈도우 적용
    num_chunks = max(1, (len(sequence) - 1) // stride + 1)
    for i in range(num_chunks):
        start = i * stride
        end = min(start + window_size, len(sequence))
        chunk = sequence[start:end]

        if len(chunk) < window_size and len(sequence) > window_size:
            chunk = sequence[-window_size:]
        spaced = _preprocess_sequence(chunk)

        inputs = tokenizer(spaced, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)

        cls = outputs.last_hidden_state[:, 0, :].cpu()
        embeddings.append(cls)

        if end >= len(sequence):
            break

    # 전체 임베딩 평균 (조각 수 만큼)
    final_embedding = torch.mean(torch.stack(embeddings), dim=0)  # shape: [1, 1024]
    return final_embedding




def make_protbert_embedding():
    input_fname = "biomedical/HI_union_Uniprot_Sequence2.txt"
    output_fname = "biomedical/HI_union_protbert_embeddings.npz"   # ".npz"로 끝나야 함

    # Uniprot ID와 서열 매핑 불러오기
    protein_dict = {}
    with open(input_fname) as f:
        for line in f:
            ensembl_id, uniprot_id, sequence = line.strip().split(" ")
            if not uniprot_id.startswith("N"):  # ID가 있는 경우에만 추가
                protein_dict[uniprot_id] = sequence

    total = len(protein_dict)
    # 모든 단백질 서열 임베딩
    embeddings = {}
    for idx, (uid, seq) in enumerate(protein_dict.items()):
        try:
            print(f"[{idx}/{total}] Embedding {uid}...")
            if len(seq) > 1024:
                emb = _sliding_embed(seq)
            else:
                emb = _embed_sequence(seq)
            embeddings[uid] = emb.numpy()
        except Exception as e:
            print(f"Failed for {uid}: {e}")


    # 임베딩을 압축된 npz 파일로 저장
    try:
        if not output_fname.endswith(".npz"):
            raise ValueError
    except ValueError:
        print(f"Output filename must end with '.npz' output_fname: {output_fname}")
    else:
        np.savez_compressed(output_fname, **embeddings)

# 임베딩 불러오기 예시
# data = np.load("/biomedical/protbert_embeddings.npz")