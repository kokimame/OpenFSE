import os
import glob, json
import freesound

client = freesound.FreesoundClient()
client.set_token("3uYSYQldiW7rON8ksZNTrUPNrs7SV8MRGUDYCDRd", "token")

with open("json/ground_truth_annotations_28_05_19.json", "r") as f:
    annons = json.load(f)

idset = []
for i, (id, label, _, duration) in enumerate(annons):
    if 5 < duration <= 20 and id not in idset:
        idset.append(id)
    if len(idset) >= 10:
        break

for id in idset:
    sound = client.get_sound(id, fields="id,previews")

    sound.retrieve_preview(f"{os.environ['HOME']}/Project/DLSOUND", f"{sound.id}.mp3")
