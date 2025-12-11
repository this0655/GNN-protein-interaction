import bidict
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

"""
This script creates node and edge data for graph neural network training.
need four input files:
- sequence_fname: mapping of ensembl_id, uniprot_id, sequence
- interaction_fname: protein-protein interaction data (ensembl_id1, ensembl_id2)
- sub_interaction_fname: STRING interaction data (ensembl_id1, ensembl_id2)
- embedding_fname: ProtBERT embeddings for each uniprot_id
"""


ensembl_to_uniprot = {}   # key: ensembl, value: uniprot
seq_to_uniprot = {}   #  key: sequence, value: uniprot(list)
idx_seq = bidict.bidict()   # key: node index, value: sequence
cache = {}
uniprot_to_seq = {}

def _make_mappings(sequence_fname):
    
    """
    Create mappings from sequence file
    1. ensembl_to_uniprot: map ensembl id to uniprot id
    2. seq_to_uniprot: map sequence to uniprot id(s)
    3. idx_seq: map node index to sequence
    """

    global ensembl_to_uniprot
    global seq_to_uniprot
    global idx_seq
    rows = 0

    with open(sequence_fname) as f:
        for line in f:
            rows += 1
            ensembl, uniprot, sequence = line.strip().split(" ")
            if not uniprot.startswith('N') or sequence != 'None':
                # 하나의 ensembl id에 해당하는 uniprot id를 저장
                ensembl_to_uniprot[ensembl] = uniprot
                
                # 하나의 sequence에 해당하는 여러 uniprot id를 리스트로 저장
                seq_to_uniprot.setdefault(sequence, set())
                seq_to_uniprot[sequence].add(uniprot)

                # sequence를 추가할 때마다, 해당 sequence를 node의 번호로 설정(추후 임베딩을 위함)
                node = len(seq_to_uniprot) - 1
                if node not in idx_seq:
                    idx_seq[node] = sequence
    
    print(f"Total number of ensembl id: {rows}")
    print(f"Except TrEMBL sequence and no uniprot id: {rows-len(ensembl_to_uniprot)}")
    print(f"Except Uniprot id has same sequence: {len(ensembl_to_uniprot) - len(seq_to_uniprot)}")
    print(f"Total number of protein: {len(seq_to_uniprot)}\n")


# 간선 만들기
def _make_uniprot_to_seq():

    """
    Create uniprot to sequence mapping using seq_to_uniprot
    1. uniprot_to_seq: map uniprot id to sequence
    """

    global uniprot_to_seq
    for seq, uni in seq_to_uniprot.items():
        for uniprot in uni:
            uniprot_to_seq[uniprot] = seq


def _ensembl_to_node_idx(ensembl):

    """
    Map ensembl id to node index using ensembl_to_uniprot, uniprot_to_seq and idx_seq
    """

    global cache
    if not uniprot_to_seq:
        _make_uniprot_to_seq()

    if ensembl in cache:
        return cache[ensembl]

    try:
        uniprot = ensembl_to_uniprot[ensembl]
        seq = uniprot_to_seq[uniprot]
        node_idx = idx_seq.inv[seq]
    except:
        return None
    else:
        return node_idx


def _make_edges(input_interaction_fname, sub_interaction_fname=None):

    """
    Make interaction edges from interaction_fname
    1. edge_index: tensor of shape (2, num_edges)
    2. edge_weight: tensor of shape (num_edges,)
    """

    edge_dict = {}
    index = []
    weight = []
    rows, no = 0, 0
    with open(input_interaction_fname, "r") as f:
        for line in f:
            interact = line.strip().split("\t")
            node1 = _ensembl_to_node_idx(interact[0])
            node2 = _ensembl_to_node_idx(interact[1])
            rows += 1

            if node1 is not None and node2 is not None:
                edge_dict[(node1, node2)] = 1.0  # use score as weight
            else:
                no += 1

    if sub_interaction_fname:
        with open(sub_interaction_fname, "r") as f:
            for line in f:
                interact = line.strip().split("\t")
                node1 = _ensembl_to_node_idx(interact[0])
                node2 = _ensembl_to_node_idx(interact[1])
                rows += 1

                if node1 is not None and node2 is not None:
                    if (node1, node2) not in edge_dict:
                        edge_dict[(node1, node2)] = float(interact[2])  # use score as weight
                else:
                    no += 1

    # 무방향 그래프이므로 양방향 간선 추가
    for (n1, n2), w in list(edge_dict.items()):
        index.append([n1, n2])
        index.append([n2, n1])
        weight.append(w)
        weight.append(w)
    
    edge_index = torch.tensor(index, dtype=torch.long).t()
    edge_weight = torch.tensor(weight, dtype=torch.float)

    print(f"간선 검사:")
    print(f"   방향성: 무방향 그래프 (양방향 간선)")
    print(f"   총 간선: {edge_index.size(1)}")
    print(f"   유니크 간선: {torch.unique(edge_index, dim=1).size(1)}")
    print(f"   Self-loop: {(edge_index[0] == edge_index[1]).sum()}")
    print(f"   안쓰인 interaction: {no}\n")   # remove interaction because of invalid ensembl id

    return edge_index, edge_weight


def _make_node_features_pca(embedding_fname, target_dim=256):

    """
    Make node features with PCA dimension reduction
    1. Load ProtBERT embeddings
    2. Create node feature matrix matching uniprot_id order
    3. Apply PCA to reduce dimensions to target_dim
    """

    # 1. ProtBERT 임베딩 로드
    df = np.load(embedding_fname)

    # 2. uniprot_id 딕셔너리와 매칭하여 노드 특성 행렬 생성
    num_nodes = len(idx_seq)
    if len(df[df.files[0]].shape) == 2:
        embedding_dim = df[df.files[0]].shape[1]
    elif len(df[df.files[0]].shape) == 1:
        embedding_dim = df[df.files[0]].shape[0]

    # ProtBERT 임베딩을 저장할 행렬 초기화
    embeddings = np.zeros((num_nodes, embedding_dim), dtype=np.float32)

    # 3. uniprot_id 순서에 맞춰 임베딩 배치
    for idx, seq in idx_seq.items():
        uniprot = next(iter(seq_to_uniprot[seq]))
        if uniprot in df.files:
            # ProtBERT 임베딩을 해당 인덱스에 저장
            embeddings[idx] = df[uniprot].squeeze()
    
    # 4. PCA 차원 축소
    pca = PCA(n_components=target_dim)
    reduced_embeddings = pca.fit_transform(embeddings)

    print(f"PCA 차원 축소: {embeddings.shape[1]} -> {target_dim}")
    explained_variance = pca.explained_variance_ratio_.sum()
    # 정보가 얼마나 보존되었는지를 보여줌
    print(f"설명된 분산 비율: {explained_variance:.4f}\n")

    return reduced_embeddings


def _make_node_features_l2(embedding_fname):

    """
    Make node features with L2 normalization
    1. Load ProtBERT embeddings
    2. Create node feature matrix matching uniprot_id order
    3. Apply L2 normalization to each node feature vector
    """
    
    # 1. ProtBERT 임베딩 로드
    df = np.load(embedding_fname)
    
    # 2. uniprot_id 딕셔너리와 매칭하여 노드 특성 행렬 생성
    num_nodes = len(idx_seq)
    if len(df[df.files[0]].shape) == 2:
        embedding_dim = df[df.files[0]].shape[1]
    elif len(df[df.files[0]].shape) == 1:
        embedding_dim = df[df.files[0]].shape[0]
    
    # 빈 특성 행렬 초기화
    node_features = np.zeros((num_nodes, embedding_dim), dtype=np.float32)
    
    # 3. uniprot_id 순서에 맞춰 임베딩 배치
    for idx, seq in idx_seq.items():
        uniprot = next(iter(seq_to_uniprot[seq]))
        if uniprot in df.files:
            embedding = df[uniprot]
            node_features[idx] = embedding.squeeze()
    
    # 4. L2 정규화 적용
    # normalize() 함수는 각 행(노드)을 L2 norm으로 나눔
    # norm='l2'는 기본값이므로 생략 가능
    normalized_features = normalize(node_features, norm='l2', axis=1)
    
    print(f"L2 정규화 적용")
    print(f"정규화 전 norm 범위: [{np.linalg.norm(node_features, axis=1).min():.4f}, "
          f"{np.linalg.norm(node_features, axis=1).max():.4f}]")
    print(f"정규화 후 norm 범위: [{np.linalg.norm(normalized_features, axis=1).min():.4f}, "
          f"{np.linalg.norm(normalized_features, axis=1).max():.4f}]")
    print(f"정규화 후 평균 norm: {np.linalg.norm(normalized_features, axis=1).mean():.6f}\n")
    
    return normalized_features


def _make_node_features(embedding_fname):

    """
    Make node features without PCA
    1. Load ProtBERT embeddings
    2. Create node feature matrix matching uniprot_id order
    """

    # 1. ProtBERT 임베딩 로드
    df = np.load(embedding_fname)

    # 2. uniprot_id 딕셔너리와 매칭하여 노드 특성 행렬 생성
    num_nodes = len(idx_seq)
    if len(df[df.files[0]].shape) == 2:
        embedding_dim = df[df.files[0]].shape[1]
    elif len(df[df.files[0]].shape) == 1:
        embedding_dim = df[df.files[0]].shape[0]

    # 빈 특성 행렬 초기화
    node_features = np.zeros((num_nodes, embedding_dim), dtype=np.float32)

    # 3. uniprot_id 순서에 맞춰 임베딩 배치
    for idx, seq in idx_seq.items():
        uniprot = next(iter(seq_to_uniprot[seq]))
        if uniprot in df.files:
            # ProtBERT 임베딩을 해당 인덱스에 저장
            embedding = df[uniprot]
            # (1, 1024) 형태를 (1024,)로 변환
            node_features[idx] = embedding.squeeze()

    return node_features


def _make_node(embedding_fname, pca=0, norm = False):

    """
    Make node tensor from embedding file
    1. Load node features with or without PCA
    2. Convert to torch tensor
    """

    if pca:
        node_features = _make_node_features_pca(embedding_fname, pca)
    elif norm:
        node_features = _make_node_features_l2(embedding_fname)
    else:
        node_features = _make_node_features(embedding_fname)
        
    node = torch.tensor(node_features, dtype=torch.float)

    print("노드 특성 검사:")
    print(f"   Shape: {node.shape}")
    print(f"   제로 노드: {(node.sum(dim=1) == 0).sum()}/{node.size(0)}")
    print(f"   범위: [{node.min():.6f}, {node.max():.6f}]")
    print(f"   평균: {node.mean():.6f}")  # PCA 적용하면 0이 정상임
    print(f"   Non-zero 비율: {(node != 0).sum().item() / node.numel():.2%}\n")

    return node


def _make_graph_data(node, edge_index, edge_weight):

    """
    Make graph data object from node, edge_index, edge_weight
    """

    graph_data = Data(
        x=node,                    # 노드 특성 (num_nodes, num_node_features)
        edge_index=edge_index,  # 간선 인덱스 (2, num_edges)
        edge_weight=edge_weight # 간선 가중치 (num_edges,)
    )
    print(f"그래프 데이터:")
    print(f"   노드 개수: {graph_data.num_nodes}")
    print(f"   간선 개수: {graph_data.num_edges}")
    print(f"   노드 특성 차원: {graph_data.num_node_features}")
    print(f"   연결된 노드: {torch.unique(graph_data.edge_index).size(0)}")
    print(f"   고립된 노드: {graph_data.num_nodes - torch.unique(graph_data.edge_index).size(0)}")

    return graph_data


def make_graph(sequence_fname, interaction_fname, sub_interaction_fname, embedding_fname, pca, norm):

    """
    Main function to create graph data
    1. Create mappings from sequence file
    2. Create edges from interaction file
    3. Create node features from embedding file
    4. Create graph data object
    """

    _make_mappings(sequence_fname)
    edge_index, edge_weight = _make_edges(interaction_fname, sub_interaction_fname)
    node = _make_node(embedding_fname, pca, norm)
    graph_data = _make_graph_data(node, edge_index, edge_weight)

    return graph_data