import csv
from collections import Counter

with open('tags_ppc_top_500.csv', 'r') as f:
    rows = []
    rows.extend(csv.reader(f))

unique_tags = set()
total_tags = []

for row in rows:
    for tag in row[1:]:
        unique_tags.add(tag)
        total_tags.append(tag)

print(f'Top Most common tags: {Counter(total_tags).most_common(5)}')
print(f'Unique tags: {len(unique_tags)}')
