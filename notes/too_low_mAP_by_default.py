# Index of closest positive and assume positive data appears consecutively from this index.
# Therefore, the farthest positive appears at index: closest_positive_index + positive_size - 1
closest_positive_index = 10
positive_size = 2
data_size = 100

spred = [0] * data_size
spred[closest_positive_index: closest_positive_index + positive_size] = list(range(1, positive_size + 1))

positions = [i for i in range(1, data_size + 1)]

ap_values = []
for index, (s, pos) in enumerate(zip(spred, positions)):
    # This condition works in the same way as the mask in the mAP computation of the MOVE
    if closest_positive_index <= index < closest_positive_index + positive_size:
        ap_values.append(s / pos)
ap = sum(ap_values) / positive_size

print(ap)