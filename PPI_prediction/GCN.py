import torch
from torch_geometric.nn import GCN
from torch_geometric.nn import GAT
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import negative_sampling
from sklearn.metrics import precision_recall_curve, roc_auc_score
import random
import numpy as np





def get_device():
    """GPU 사용 가능 여부 확인 및 디바이스 반환"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"GPU 사용: {torch.cuda.get_device_name(0)}")
        print(f"메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
        return device
    else:
        device = torch.device('cpu')
        print("GPU 없음 - CPU 사용\n")
        return None



# 1. 데이터 분할
def spilt_data(graph_data):
    train_data, val_data, test_data = RandomLinkSplit(
        num_val=0.1, num_test=0.1, is_undirected=True
    )(graph_data)

    device = get_device()
    if device is not None:
        train_data = train_data.to(device)
        val_data = val_data.to(device)
        test_data = test_data.to(device)
    
    print(f"Train: {train_data.edge_index.size(1)} edges")
    print(f"Val:   {val_data.edge_label.size(0)} edges")
    print(f"Test:  {test_data.edge_label.size(0)} edges\n")

    return train_data, val_data, test_data



# 2. 모델 & 옵티마이저
def set_model(in_ch, hidden, layers, out, running_rate, dropout, decay, model_type='GCN'):
    if model_type == 'GCN':
        model = GCN(
            in_channels = in_ch, 
            hidden_channels = hidden, 
            num_layers = layers, 
            out_channels = out,
            dropout = dropout,
            act='relu'
        )
    elif model_type == 'GAT':
        model = GAT(
            in_channels = in_ch, 
            hidden_channels = hidden, 
            num_layers = layers, 
            out_channels = out,
            dropout = dropout,
            heads = 4,
            act='elu'
        )

    device = get_device()
    if model is not None:
        model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=running_rate, 
        weight_decay=decay
    )

    return model, optimizer



# 3. 학습 & 평가
def train(model, optimizer, train_data):
    model.train()
    optimizer.zero_grad()
    
    z = model(train_data.x, train_data.edge_index)
    pos = train_data.edge_index   # 실제 연결된 엣지를 양성으로
    neg = negative_sampling(pos, train_data.num_nodes, pos.size(1))
    # 존재하지 않는 엣지에서 양성과 같은 수만큼 뽑음
    
    pos_loss = -torch.log((z[pos[0]] * z[pos[1]]).sum(1).sigmoid() + 1e-15).mean()
    neg_loss = -torch.log(1 - (z[neg[0]] * z[neg[1]]).sum(1).sigmoid() + 1e-15).mean()
    # (z[pos[0]] * z[pos[1]]).sum(1): 점곱, 점수가 클수록 연결될 확률이 높다고 가정
    # pos_loss = -log(sigmoid(pos_score)) => 로지스틱 손실로 양성은 1에 가깝게, 음성은 0에 가깝게.
    
    loss = pos_loss + neg_loss
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, train_data, return_dict = False):   # data를 validation data나 test data로 사용함.
    model.eval()
    z = model(train_data.x, train_data.edge_index)
    pred = (z[data.edge_label_index[0]] * z[data.edge_label_index[1]]).sum(1).sigmoid()
    label = data.edge_label.cpu()
    score = pred.cpu()
    if not return_dict:
        return roc_auc_score(label, score)
    
    label = data.edge_label.cpu().numpy()
    score = pred.cpu().numpy()
    precisions, recalls, thresholds = precision_recall_curve(label, score)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_f1_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_f1_idx]
    best_threshold = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else 0.5
    return {
        'auc': roc_auc_score(label, score),
        'f1': best_f1,
        'accuracy': ((score >= best_threshold) == label).mean()
    }



# 4. 하이퍼파라미터 튜닝
def hyperparameter_tuning(in_channel):
    while True:
        hidden_channels = random.choice([1024, 512, 256])
        num_layers = random.choice([1, 2, 3])
        out_channels = random.choice([256, 128])
        running_rate = random.choice([0.001, 0.0005, 0.0002])
        dropout = random.choice([0.2, 0.3, 0.4])
        decay = random.choice([1e-3, 5e-4, 1e-4, 1e-5])
        if hidden_channels > out_channels:
            print(f"\nhidden channels {hidden_channels}")
            print(f"layer numbers {num_layers}")
            print(f"out channels {out_channels}")
            print(f"running rate {running_rate}")
            print(f"dropout rate {dropout}")
            print(f"weight decay {decay}\n")
            return hidden_channels, num_layers, out_channels, running_rate, dropout, decay