import os
import csv
import glob
import json
import shutil
import requests
import freesound

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


client = freesound.FreesoundClient()
client.set_token('3uYSYQldiW7rON8ksZNTrUPNrs7SV8MRGUDYCDRd', 'token')

IDLIMIT = 0
DATASET_NAME = 'audio_200k'
DOWNLOAD_DIR = f'/media/kokimame/Work_A_1TB/Project/Master_Files/{DATASET_NAME}'
if not os.path.exists(DOWNLOAD_DIR):
    os.makedirs(DOWNLOAD_DIR)

annons = []
with open('../data/tags_ppc_top_500.csv', 'r') as f:
    annons.extend(csv.reader(f))
ids = [annon[0] for annon in annons[1:]] # Remove header

total_download = 0
start_chunk = 797
for ith, id_chunk in enumerate(chunks(ids, 50)):

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
            sound.retrieve_preview(DOWNLOAD_DIR, f'{str(sound.id)}.mp3')
            total_download += 1
            print(f'{i+1:4d}\t|\t{sound.id} - {sound.name}')
        except KeyError as e:
            print(f'Unexpected sound caught from freesound id:{sound.id}')
        except requests.ConnectionError:
            print(f'Connection Error: Skip a sound')
        except Exception as e:
            if '404 Not Found' in str(e):
                continue
            else:
                raise e
