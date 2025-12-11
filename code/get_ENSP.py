import requests

ensp_ensg = {}

ensp_url = 'https://string-db.org/api/tsv/get_string_ids'

fname = 'biomedical/union_protein.txt'

with open(fname) as f:
    for line in f:
        ensg_id = line.strip()
        param = {
            'identifiers': ensg_id,
            'species': 9606,
            'limit': 1,
            'echo_query': 1
        }
        response = requests.get(ensp_url, params=param)
        if response.status_code == 200:
            data = response.text.strip().split("\t")
            if len(data) > 8:
              ensp_id = data[8].lstrip('9606.')
              print(ensp_id, ensg_id)
              ensp_ensg[ensp_id] = ensg_id
            else:
              print(f"No data found for {ensg_id}")
              ensp_ensg[ensp_id] = 'None'
        else:
            print(f"Error fetching data for {ensg_id}: {response.status_code}")

with open('biomedical/ensp_ensg1.txt', 'w') as f:
    for ensp_id, ensg_id in ensp_ensg.items():
        f.write(f"{ensp_id}\t{ensg_id}\n")
        f.flush()