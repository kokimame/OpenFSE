import os
import glob, json
import freesound

client = freesound.FreesoundClient()
client.set_token("3uYSYQldiW7rON8ksZNTrUPNrs7SV8MRGUDYCDRd", "token")
DLTOP = f"{os.environ['HOME']}/Project/Master_Files/audio"
if not os.path.exists(DLTOP):
    os.makedirs(DLTOP)

with open("json/ground_truth_annotations_28_05_19.json", "r") as f:
    annons = json.load(f)

idset = []
for i, (id, label, _, duration) in enumerate(annons):
    if 5 < duration <= 20 and id not in idset:
        idset.append(id)
    if len(idset) >= 100:
        break

for id in idset:
    sound = client.get_sound(id, fields="id,previews")
    # This is not efficient. Use a filter
    sound.retrieve_preview(DLTOP, f"{sound.id}.mp3")
