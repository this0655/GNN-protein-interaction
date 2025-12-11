import os
os.chdir(r"c:\\Users\\this0\\pyprogramming\\biomedical")

"""
Remove dupicated ensembl id with interaction file
input: interaction tsv file
output: unique protein list file
"""


input_fname = "HI-union.tsv"
output_fname = "union_protein.txt"

proteins = set()
with open(input_fname, "r", encoding="utf-8") as f:
    for line in f:
        words = line.rstrip("\n").split("\t")
        for w in words:
            if w not in proteins:
                proteins.add(w)

sort_proteins = sorted(list(proteins))

with open(output_fname, "w", encoding="utf-8") as f:
    for p in sort_proteins:
        f.write(p + "\n")
