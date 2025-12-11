from transformers import T5Tokenizer, T5EncoderModel
import torch
import numpy as np
import os

"""
Make ProtT5-XL-U50 embeddings for protein sequences
Input: text file with uniprot id and sequence
Output: npz file with uniprot id and ProtT5 embedding

ProtT5-XL-U50:
    - Model: Rostlab/prot_t5_xl_uniref50
    - Parameters: ~3B
    - Dimensions: 1024
    - Trained on UniRef50
"""

# !pip install transformers[sentencepiece]

protein_dict = {}
embeddings = {}
processed = set()
tokenizer, model, device, output_fname = None, None, None, None



def _load_prott5_model():
    """
    ProtT5-XL-U50 모델 로드
    
    Returns:
        tokenizer, model, device
    """
    global tokenizer, model, device, output_fname
    
    model_name = "Rostlab/prot_t5_xl_uniref50"
    if output_fname is None:
        output_fname = "biomedical/HI_union_protT5_xl_embeddings.npz"
    
    print(f"Loading model: {model_name}")
    print(f"Embedding dimension: 1024")
    
    # Tokenizer 로드
    tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
    
    # Model 로드 (encoder만 사용)
    model = T5EncoderModel.from_pretrained(model_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = model.to(device)
    model.eval()
    


@torch.no_grad()
def embed_sequence(sequence, tokenizer, model, device):
    """
    단일 시퀀스 임베딩
    
    ProtT5는 공백으로 구분된 residue를 입력받습니다.
    예: "M K L L" (각 아미노산 사이 공백)
    
    Args:
        sequence: 단백질 서열 (공백 없음)
    
    Returns:
        embedding: [1024] numpy array (전체 서열의 평균)
    """
    # 공백 제거 및 공백으로 구분
    sequence = sequence.replace(" ", "")
    sequence_with_spaces = " ".join(list(sequence))
    
    # ProtT5는 최대 길이 제한이 있음 (보통 1000 residues)
    if len(sequence) > 1000:
        print(f"  ⚠ Sequence too long ({len(sequence)}), truncating to 1000")
        sequence = sequence[:1000]
        sequence_with_spaces = " ".join(list(sequence))
    
    # Tokenize
    inputs = tokenizer(
        sequence_with_spaces,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Forward pass
    outputs = model(**inputs)
    
    # Mean pooling over sequence length
    # outputs.last_hidden_state: [1, seq_len, 1024]
    embedding = outputs.last_hidden_state.mean(dim=1)  # [1, 1024]
    
    return embedding.cpu().numpy().squeeze()  # [1024]


@torch.no_grad()
def embed_sequence_per_residue(sequence, tokenizer, model, device):
    """
    Per-residue 임베딩 (선택적)
    
    Returns:
        embeddings: [seq_len, 1024] numpy array
    """
    sequence = sequence.replace(" ", "")
    sequence_with_spaces = " ".join(list(sequence))
    
    if len(sequence) > 1000:
        sequence = sequence[:1000]
        sequence_with_spaces = " ".join(list(sequence))
    
    inputs = tokenizer(
        sequence_with_spaces,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    outputs = model(**inputs)
    
    # Per-residue embeddings (special tokens 제외)
    # [1, seq_len, 1024] -> [seq_len, 1024]
    per_residue_emb = outputs.last_hidden_state[0]  # Remove batch dimension
    
    return per_residue_emb.cpu().numpy()


@torch.no_grad()
def sliding_embed(sequence, tokenizer, model, device, window_size=900, stride=450):
    """
    긴 시퀀스 처리 (슬라이딩 윈도우)
    
    Args:
        sequence: 단백질 서열
        window_size: 윈도우 크기 (ProtT5 최대 ~1000)
        stride: 슬라이딩 간격
    
    Returns:
        embedding: [1024] numpy array (여러 청크의 평균)
    """
    sequence = sequence.replace(" ", "")
    embeddings = []
    
    # 청크 개수 계산
    num_chunks = max(1, (len(sequence) - window_size) // stride + 1)
    
    for i in range(num_chunks):
        start = i * stride
        end = min(start + window_size, len(sequence))
        chunk = sequence[start:end]
        
        # 마지막 청크가 너무 짧으면 끝에서부터
        if len(chunk) < window_size and len(sequence) > window_size:
            chunk = sequence[-window_size:]
        
        # 공백으로 구분
        chunk_with_spaces = " ".join(list(chunk))
        
        # Tokenize
        inputs = tokenizer(
            chunk_with_spaces,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Forward pass
        outputs = model(**inputs)
        chunk_embedding = outputs.last_hidden_state.mean(dim=1).cpu()
        embeddings.append(chunk_embedding)
        
        # 마지막 청크까지 처리했으면 종료
        if end >= len(sequence):
            break
    
    # 여러 청크의 평균
    final_embedding = torch.mean(torch.stack(embeddings), dim=0)  # [1, 1024]
    return final_embedding.numpy().squeeze()  # [1024]


# ===== 데이터 로드 =====
def _load_data(input_fname):
    """
    Uniprot ID와 서열 매핑
    
    Format: <index> <uniprot_id> <sequence>
    """
    global protein_dict
    
    protein_dict = {}
    
    with open(input_fname, 'r') as f:
        for line in f:
            parts = line.strip().split(" ")
            if len(parts) >= 3:
                uniprot_id = parts[1]
                sequence = parts[2]
                
                # "N"으로 시작하는 ID 제외 (노이즈 데이터)
                if not uniprot_id.startswith("N"):
                    protein_dict[uniprot_id] = sequence
    
    print(f"Total proteins: {len(protein_dict)}")
    
    # 길이 통계
    lengths = [len(seq) for seq in protein_dict.values()]
    print(f"Sequence length: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.1f}")
    
    return protein_dict


# ===== Checkpoint 관리 =====
def _load_checkpoint():
    """
    이미 처리한 임베딩 로드
    """
    global embeddings, processed, output_fname
    
    embeddings = {}
    processed = set()
    
    if os.path.exists(output_fname):
        print(f"\n✓ Found checkpoint: {output_fname}")
        checkpoint = np.load(output_fname)
        embeddings = {k: checkpoint[k] for k in checkpoint.files}
        processed = set(embeddings.keys())
        print(f"  Resuming from {len(processed)} proteins")
    else:
        print(f"\n✓ Starting fresh (no checkpoint found)")
    
    return embeddings, processed


def _save_checkpoint():
    """
    임베딩 저장
    """
    global embeddings, output_fname
    
    np.savez_compressed(output_fname, **embeddings)
    print(f"  ✓ Saved {len(embeddings)} embeddings to: {output_fname}")



def _make_embeddings(save_every=10):
    """
    모든 단백질 임베딩 생성
    
    Args:
        save_every: N개마다 중간 저장
    """
    global protein_dict, embeddings, processed, tokenizer, model, device
    
    _load_checkpoint()
    
    total = len(protein_dict)
    remaining = total - len(processed)
    
    print(f"\n{'='*60}")
    print(f"Starting embedding generation")
    print(f"  Total proteins: {total}")
    print(f"  Already processed: {len(processed)}")
    print(f"  Remaining: {remaining}")
    print(f"{'='*60}\n")
    
    for idx, (uid, seq) in enumerate(protein_dict.items(), 1):
        try:
            # 이미 처리한 단백질 건너뛰기
            if uid in processed:
                continue
            
            print(f"[{idx}/{total}] Embedding {uid} (length: {len(seq)})...")
            
            # 길이에 따라 임베딩 방법 선택
            if len(seq) > 1000:
                print(f"  → Using sliding window (seq length: {len(seq)})")
                emb = sliding_embed(seq, tokenizer, model, device)
            else:
                emb = embed_sequence(seq, tokenizer, model, device)
            
            embeddings[uid] = emb  # [1024]
            processed.add(uid)
            
            print(f"  ✓ Embedding shape: {emb.shape}, range: [{emb.min():.4f}, {emb.max():.4f}]")
            
        except Exception as e:
            print(f"  ✗ Failed for {uid}: {e}")
            continue
        
        # 주기적 저장
        if idx % save_every == 0:
            print(f"\n--- Checkpoint {idx // save_every} ---")
            _save_checkpoint()
            print()
    
    # 최종 저장
    print(f"\n{'='*60}")
    print("Final save...")
    _save_checkpoint()
    print(f"{'='*60}\n")
    
    return embeddings


# ===== 결과 확인 =====
def _verify_embeddings(num_samples=3):
    """
    생성된 임베딩 확인
    """
    global output_fname
    
    if not os.path.exists(output_fname):
        print(f"File not found: {output_fname}")
        return

    data = np.load(output_fname)
    
    print(f"\nTotal proteins: {len(data.files)}")
    
    print(f"\nSample embeddings (first {num_samples}):")
    for i, uid in enumerate(data.files[:num_samples]):
        emb = data[uid]
        print(f"\n  [{i+1}] UID: {uid}")
        print(f"      Shape: {emb.shape}")
        print(f"      Range: [{emb.min():.4f}, {emb.max():.4f}]")
        print(f"      Mean: {emb.mean():.4f}, Std: {emb.std():.4f}")


# ===== 메인 실행 함수 =====
def start_embedding(input_fname = None, fname = None):
    """
    ProtT5 임베딩 파이프라인 실행
    
    Args:
        input_fname: 입력 파일 경로
    """
    if input_fname is None:
        input_fname="biomedical/HI_union_Uniprot_Sequence2.txt"
    global output_fname
    output_fname = fname

    # 1. 모델 로드
    print("Loading ProtT5 model...")
    _load_prott5_model()
    
    # 2. 데이터 로드
    print(f"\nLoading protein sequences...")
    _load_data(input_fname)
    
    # 3. 임베딩 생성
    print(f"\nGenerating embeddings...")
    _make_embeddings(save_every=10)
    
    # 4. 결과 확인
    print(f"\nVerifying results...")
    _verify_embeddings()
    
    print(f"\n{'='*60}")
    print("✓ All done!")
    print(f"{'='*60}\n")


# ===== 메인 실행 =====
if __name__ == "__main__":
    start_embedding()