import csv
import matplotlib.pyplot as plt
import collections

hard_indices = []
with open('../hard_indices.csv', 'r') as f:
    hard_indices.extend(csv.reader(f))

pex = [int(i[0]) for i in hard_indices]
pC = collections.Counter(pex)
pid = list(set(pex))

nex = [int(i[1]) for i in hard_indices]
nC = collections.Counter(nex)
nid = list(set(nex))

indices = list(set(pex + nex))

plt.subplot(2, 1, 1)
plt.title('Farthest Positives')
plt.bar(pC.keys(), pC.values())

plt.subplot(2, 1, 2)
plt.title('Closest Negatives')
plt.bar(nC.keys(), nC.values())
plt.show()

