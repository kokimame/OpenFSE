import json
import matplotlib.pyplot as plt

JSONFILE = 'ground_truth_annotations_28_05_19.json'

with open(JSONFILE, 'r') as f:
    data = json.load(f)

lookup = {}
for sound, label, _, _ in data:
    labels = lookup.get(sound, [])
    labels.append(label)
    lookup[sound] = labels

vals = [len(v) for k, v in lookup.items()]

plt.hist(vals)
plt.xticks(range(0, max(vals)))
plt.ylabel('# of sounds')
plt.xlabel('# of labels assigned to a sound')
plt.title(f'FS AudioSet annotation data. Size: {len(data)}')
plt.show()