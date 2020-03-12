import os
import glob, json
import freesound

client = freesound.FreesoundClient()
client.set_token("3uYSYQldiW7rON8ksZNTrUPNrs7SV8MRGUDYCDRd", "token")
DLTOP = f"{os.environ['HOME']}/Project/Master_Files/audio_v2"
if not os.path.exists(DLTOP):
    os.makedirs(DLTOP)

with open("json/ground_truth_annotations_28_05_19.json", "r") as f:
    annons = json.load(f)

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

idset = []
for i, (id, label, _, duration) in enumerate(annons):
    if 5 < duration <= 20 and str(id) not in idset:
        idset.append(str(id))
    if len(idset) >= 10000:
        break


total_download = 1
for id_chunk in chunks(idset, 50):
    print(
        '\n\nAPI CALL\n\n'
    )
    results_pager = client.text_search(
        filter=f'id:({" OR ".join(id_chunk)})',
        page_size=50,
        fields="id,previews,name"
    )

    for i, sound in enumerate(results_pager):
        print(total_download, ' ... ', i + 1, sound.id, sound.name)
        sound.retrieve_preview(DLTOP, f"{sound.id}.mp3")
        total_download += 1
