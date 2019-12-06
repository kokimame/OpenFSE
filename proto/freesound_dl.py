import os
import glob, json
import freesound

client = freesound.FreesoundClient()
client.set_token("3uYSYQldiW7rON8ksZNTrUPNrs7SV8MRGUDYCDRd", "token")

results = client.text_search(query="106737",fields="id,previews")

print(results)

with open("json/ground_truth_annotations_28_05_19.json", "r") as f:
    annons = json.load(f)
    print(len(annons))

for sound in results:
    sound.retrieve_preview("DLSOUND", f"{sound.id}.mp3")
