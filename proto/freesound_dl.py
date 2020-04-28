import os
import glob, json
import freesound
client = freesound.FreesoundClient()
client.set_token("3uYSYQldiW7rON8ksZNTrUPNrs7SV8MRGUDYCDRd", "token")
IDLIMIT = 0
DATASET_NAME = 'audio_tagged'
DOWNLOAD_DIR = f"{os.environ['HOME']}/Project/Master_Files/{DATASET_NAME}"
if not os.path.exists(DOWNLOAD_DIR):
    os.makedirs(DOWNLOAD_DIR)

with open("json/ground_truth_annotations_28_05_19.json", "r") as f:
    annons = json.load(f)

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

label_lookup = {}
for id, label, conf, duration in annons:
    try:
        _ = label_lookup[str(id)]
    except:
        label = label.replace('/', '_')
        if 5 < duration <= 20:
            label_lookup[str(id)] = label
        if IDLIMIT and len(label_lookup.keys()) >= IDLIMIT:
            break


total_download = 0
start_chunk = 126
for ith, id_chunk in enumerate(chunks(list(label_lookup.keys()), 50)):

    if ith < start_chunk: continue
    print(
        f'- + - + - + - + - + - + -\nFreesound API CALL ({ith}-th chunk)\n'
    )
    results_pager = client.text_search(
        filter=f'id:({" OR ".join(id_chunk)})',
        page_size=50,
        fields="id,previews,name"
    )

    for i, sound in enumerate(results_pager):
        try:
            dir = os.path.join(DOWNLOAD_DIR, label_lookup[str(sound.id)])
            if not os.path.exists(dir):
                os.makedirs(dir)

            sound.retrieve_preview(dir, f"{str(sound.id)}.mp3")
            total_download += 1
            print(f'Total downloads ... {i:4d}\t|\t{sound.id} - {sound.name}')
        except KeyError as e:
            print(f'Unexpected sound caught from freesound id:{sound.id}')
        except Exception as e:
            if '404 Not Found' in str(e):
                continue
            else:
                raise e
