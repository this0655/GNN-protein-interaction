from transformers import EsmTokenizer, EsmModel
import torch
import numpy as np
import os

"""
Make ESM1b-650M or ESM2-650M embeddings for protein sequences
Input: text file with uniprot id and sequence
Output: npz file with uniprot id and ESM1b, ESM2 embedding

Model1: ESM1b-650M
    Parameters: 650M
    Dimensions: 1280
    Trained on UniRef50

Model2: ESM2-650M
    Parameters: 650M
    Dimensions: 1280
    Trained on UniRef50
"""


protein_dict, embeddings = {}, {}
processed = set()
tokenizer, model, device, output_fname = None, None, None, None


# 모델 로드
def _load_esm_model(model_type):
    global tokenizer, model, device, output_fname

    if model_type == 'esm1b':
        model_name = "facebook/esm1b_t33_650M_UR50S"
        if not output_fname:
            output_fname = "biomedical/HI_union_esm1b_650M_embeddings.npz"
    elif model_type == 'esm2':
        model_name = "facebook/esm2_t33_650M_UR50D"
        if not output_fname:
            output_fname = "biomedical/HI_union_esm2_650M_embeddings.npz"
    print(f"Loading model: {model_name}")

    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()


@torch.no_grad()
def _embed_sequence(sequence):
    """
    ESM2는 공백 없이 입력
    """
    inputs = tokenizer(sequence, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :]  # [1, 480]
    return embedding.cpu()


@torch.no_grad()
def _sliding_embed(sequence, window_size=1022, stride=512):
    """
    긴 시퀀스 처리 (슬라이딩 윈도우)
    ESM의 최대 길이는 1024 (특수 토큰 제외 시 1022)
    """
    sequence = sequence.replace(" ", "")
    embeddings = []

    num_chunks = max(1, (len(sequence) - 1) // stride + 1)
    for i in range(num_chunks):
        start = i * stride
        end = min(start + window_size, len(sequence))
        chunk = sequence[start:end]

        if len(chunk) < window_size and len(sequence) > window_size:
            chunk = sequence[-window_size:]

        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)

        cls = outputs.last_hidden_state[:, 0, :].cpu()
        embeddings.append(cls)

        if end >= len(sequence):
            break

    final_embedding = torch.mean(torch.stack(embeddings), dim=0)  # [1, 480]
    return final_embedding



# Uniprot ID와 서열 매핑
def _make_uniprot_id_list(input_fname):
    global protein_dict
    with open(input_fname) as f:
        for line in f:
            _, uniprot_id, sequence = line.strip().split(" ")
            if not uniprot_id.startswith("N"):
                protein_dict[uniprot_id] = sequence
    print(f"Total proteins: {len(protein_dict)}\n")



# 이미 처리한 것 로드
def _continue_from_checkpoint(output_fname):
    global embeddings
    global processed
    if os.path.exists(output_fname):
        checkpoint = np.load(output_fname)
        embeddings = {k: checkpoint[k] for k in checkpoint.files}
        processed = set(embeddings.keys())
        print(f"✓ Resuming from {len(processed)} proteins")



# 임베딩 생성
def make_embedding(protein_dict):
    for idx, (uid, seq) in enumerate(protein_dict.items()):
        try:
            print(f"[{idx+1}/{len(protein_dict)}] Embedding {uid}...")

            # 이미 처리한 단백질 건너뛰기
            if uid in processed:
                print(f"  - Already processed {uid}, skipping.")
                continue

            if len(seq) > 1022:
                emb = _sliding_embed(seq)
            else:
                emb = _embed_sequence(seq)
            
            embeddings[uid] = emb.numpy().squeeze()
        except Exception as e:
            print(f"Failed for {uid}: {e}")

        if (idx+1) % 10 == 0:
            print(f"Save {(idx // 10) + 1} times...")
            np.savez_compressed(output_fname, **embeddings)
            print(f"\n✓ Saved embeddings to: {output_fname}")

    np.savez_compressed(output_fname, **embeddings)
    print(f"\n✓ Saved embeddings to: {output_fname}")

    # 확인
    data = np.load(output_fname)
    sample_uid = data.files[0]
    sample_emb = data[sample_uid]
    print(f"\nSample embedding:")
    print(f"  UID: {sample_uid}")
    print(f"  Shape: {sample_emb.shape}")  # (480,)
    print(f"  Range: [{sample_emb.min():.4f}, {sample_emb.max():.4f}]")



def start_embedding(model_type, input_fname = None):
    if input_fname is None:
        input_fname = "biomedical/HI_union_Uniprot_Sequence2.txt"
    _make_uniprot_id_list(input_fname)
    _continue_from_checkpoint(output_fname)
    _load_esm_model(model_type)
    make_embedding(protein_dict)

