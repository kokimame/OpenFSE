import torch

t = torch.tensor([
    [0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0],
    [0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1],
    [0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1],
])

matrix = t.tolist()
output = []
for row in matrix:
    output.append([i for i, value in enumerate(row) if value == 1])

print(output)
# [[1, 5, 6, 7, 10, 11], [1, 5, 6, 9, 10, 11], [3, 4, 6, 10, 11, 12], [1, 3, 6, 7, 8, 10, 11], [1, 5, 6, 7, 10, 11, 12]]

outt = torch.tensor(output)
print(outt.min().tolist())