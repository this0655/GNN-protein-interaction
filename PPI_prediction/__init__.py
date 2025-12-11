"""
Biomedical Analysis Package
단백질 상호작용(PPI) 예측을 위한 데이터 수집, 전처리, 임베딩 및 GCN 모델링 패키지입니다.
"""

__version__ = "0.1.0"

# 1. 데이터 수집 및 전처리 모듈
from .Interaction_to_Protein import make_ensg_id
from .Ensembl_to_Sequence import uniprot_request
from .get_ENSP import make_ensp_ensg
from .STRING_interaction import make_interaction_file

# 2. 임베딩 생성 모듈 (함수 이름 충돌 방지를 위해 alias 사용)
from .ProtBERT_embedding import make_protbert_embedding
from .ProtT5_embedding import start_embedding as make_prott5_embedding
from .ESM_embedding import start_embedding as make_esm_embedding

# 3. 그래프 데이터 구축 모듈
from .make_node_edge_graph import make_graph

# 4. GCN 모델링 및 학습 모듈
from .GCN import (
    spilt_data,
    set_model,
    train,
    evaluate,
    hyperparameter_tuning,
)

__all__ = [
    # Data Processing
    "make_ensg_id",
    "uniprot_request",
    "make_ensp_ensg",
    "make_interaction_file",
    
    # Embeddings
    "make_protbert_embedding",
    "make_prott5_embedding",  # start_embedding -> make_prott5_embedding
    "make_esm_embedding",     # start_embedding -> make_esm_embedding
    
    # Graph Construction
    "make_graph",
    
    # Model Training
    "spilt_data",
    "set_model",
    "train",
    "evaluate",
    "hyperparameter_tuning",
]