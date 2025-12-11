import requests
import json
import random

ensp_ensg = {}
with open('biomedical/ensp_ensg.txt') as f:
    for line in f:
        ensp_id, ensg_id = line.strip().split("\t")
        ensp_ensg[ensp_id] = ensg_id

protein_ids = []
with open('biomedical/union_protein.txt') as f:
    for line in f:
        ensp_id = line.strip()
        protein_ids.append(ensp_id)

interaction_data = {}

fname = 'biomedical/union_protein.txt'

network_url = 'https://string-db.org/api/json/network'

# 순차 탐색
print("Starting sequential sampling...")
for i in range(0, len(protein_ids), 100):
    s = i
    e = min(i + 2000, len(protein_ids))
    load = {
        'identifiers': "%0d".join(protein_ids[s:e]), 
        'species': 9606,
        'network_type': 'physical',
        'required_score': 700,
        'caller_identity': 'pyprogramming_test'
    }

    try:
        print("Requesting STRING database...")
        response = requests.post(network_url, data=load)

        if response.status_code == 200:
            network_data = response.json()
            print(f"Received {len(network_data)} interactions.")
        else:
            raise Exception()
        
    except Exception:
        print(f"Request failed with status code {response.status_code}")
    else:
        response_json = response.json()
    
    # interaction 저장
    for interaction in response_json:
        ensp1 = interaction['stringId_A'].lstrip('9606.')
        ensp2 = interaction['stringId_B'].lstrip('9606.')
        if interaction['escore'] > 0 or interaction['dscore'] > 0:
            ensg1 = ensp_ensg.get(ensp1, 'None')
            ensg2 = ensp_ensg.get(ensp2, 'None')

        if ensg1 != 'None' and ensg2 != 'None':
            [ensg1, ensg2] = sorted([ensg1, ensg2])
            key = (ensg1, ensg2)
            if key not in interaction_data:
                interaction_data[key] = interaction['score']

    print(f"Processed proteins {s} to {e}")
    print(f"Total interactions: {len(interaction_data)}")
 


 


# 랜덤 탐색
print("Starting random sampling...")
for _ in range(1000):
    protein_ids_sample = random.sample(protein_ids, 2000)
    load = {
        'identifiers': "%0d".join(protein_ids_sample), 
        'species': 9606,
        'network_type': 'physical',
        'required_score': 700,
        'caller_identity': 'pyprogramming_test'
    }

    try:
        print("Requesting STRING database...")
        response = requests.post(network_url, data=load)

        if response.status_code == 200:
            network_data = response.json()
            print(f"Received {len(network_data)} interactions.")
        else:
            raise Exception()
        
    except Exception as e:
        print(f"Request failed with status code {response.status_code}")
    else:
        response_json = response.json()
    
    # interaction 저장
    for interaction in response_json:
        ensp1 = interaction['stringId_A'].lstrip('9606.')
        ensp2 = interaction['stringId_B'].lstrip('9606.')
        if interaction['escore'] > 0 or interaction['dscore'] > 0:
            ensg1 = ensp_ensg.get(ensp1, 'None')
            ensg2 = ensp_ensg.get(ensp2, 'None')

        if ensg1 != 'None' and ensg2 != 'None':
            [ensg1, ensg2] = sorted([ensg1, ensg2])
            key = (ensg1, ensg2)
            if key not in interaction_data:
                interaction_data[key] = interaction['score']

    print(f"Total interactions: {len(interaction_data)}")


# 정렬 후 저장
interaction_data = dict(sorted(interaction_data.items()))

with open('STRING_interactions.txt', 'w') as f:
    for (ensg1, ensg2), score in interaction_data.items():
        f.write(f"{ensg1}\t{ensg2}\t{score}\n")
    