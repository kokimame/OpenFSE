r"""A simple demonstration of running VGGish in inference mode.
This is intended as a toy example that demonstrates how the various building
blocks (feature extraction, model definition and loading, postprocessing) work
together in an inference context.
A WAV file (assumed to contain signed 16-bit PCM samples) is read in, converted
into log mel spectrogram examples, fed into VGGish, the raw embedding output is
whitened and quantized, and the postprocessed embeddings are optionally written
in a SequenceExample to a TFRecord file (using the same format as the embedding
features released in AudioSet).
Usage:
  # Run a WAV file through the model and print the embeddings. The model
  # checkpoint is loaded from vggish_model.ckpt and the PCA parameters are
  # loaded from vggish_pca_params.npz in the current directory.
  $ python vggish_inference_demo.py --wav_file /path/to/a/wav/file
  # Run a WAV file through the model and also write the embeddings to
  # a TFRecord file. The model checkpoint and PCA parameters are explicitly
  # passed in as well.
  $ python vggish_inference_demo.py --wav_file /path/to/a/wav/file \
                                    --tfrecord_file /path/to/tfrecord/file \
                                    --checkpoint /path/to/model/checkpoint \
                                    --pca_params /path/to/pca/params
  # Run a built-in input (a sine wav) through the model and print the
  # embeddings. Associated model files are read from the current directory.
  $ python vggish_inference_demo.py
"""

from __future__ import print_function

import os
import numpy as np
import six
import soundfile as sf
import glob
import json
from pathlib import Path
import csv
from tqdm import tqdm

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim

flags = tf.app.flags

flags.DEFINE_string(
    'checkpoint', '/media/kokimame/Work_A_1TB/Project/Master_Files/audioset_vggish/vggish_model.ckpt',
    'Path to the VGGish checkpoint file.')

flags.DEFINE_string(
    'pca_params', '/media/kokimame/Work_A_1TB/Project/Master_Files/audioset_vggish/vggish_pca_params.npz',
    'Path to the VGGish PCA parameters file.')

flags.DEFINE_string(
    'tfrecord_file', None,
    'Path to a TFRecord file where embeddings will be written.')

FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.ERROR)
ONTROLOGY = '/home/kokimame/Dropbox/OpenFSE/json/ontology.json'
ROOTDIR = f'/media/kokimame/Work_A_1TB/Project/Master_Files'
OUTPUTDIR = f'{ROOTDIR}/projector'
AUDIO_CHUNKS = f'{OUTPUTDIR}/audio'

def main(_):
    # # In this simple example, we run the examples from a single audio file through
    # # the model. If none is provided, we generate a synthetic input.
    # if FLAGS.wav_file:
    #   wav_file = FLAGS.wav_file
    # else:
    #   # Write a WAV of a sine wav into an in-memory file object.
    #   num_secs = 5
    #   freq = 1000
    #   sr = 44100
    #   t = np.linspace(0, num_secs, int(num_secs * sr))
    #   x = np.sin(2 * np.pi * freq * t)
    #   # Convert to signed 16-bit samples.
    #   samples = np.clip(x * 32768, -32768, 32767).astype(np.int16)
    #   wav_file = six.BytesIO()
    #   soundfile.write(wav_file, samples, sr, format='WAV', subtype='PCM_16')
    #   wav_file.seek(0)

    ontology_lookup = {}
    with open(ONTROLOGY, 'r') as f:
        label_json = json.load(f)
    for entry in label_json:
        label_id = entry['id'].replace('/', '_')
        assert label_id not in ontology_lookup.keys()
        ontology_lookup[label_id] = entry
    wav_paths = glob.glob(os.path.join(AUDIO_CHUNKS, '*', '*.wav'))

    # Prepare a postprocessor to munge the model embeddings.
    pproc = vggish_postprocess.Postprocessor(FLAGS.pca_params)

    audio_tsv = []
    label_tsv = []
    emb_tsv = []
    for wavfile in tqdm(wav_paths):
        label = Path(Path(wavfile).parent).stem
        filename = Path(wavfile).name
        examples_batch = vggish_input.wavfile_to_examples(wavfile)

        with tf.Graph().as_default(), tf.Session() as sess:
            # Define the model in inference mode, load the checkpoint, and
            # locate input and output tensors.
            vggish_slim.define_vggish_slim(training=False)
            vggish_slim.load_vggish_slim_checkpoint(sess, FLAGS.checkpoint)
            features_tensor = sess.graph.get_tensor_by_name(
                vggish_params.INPUT_TENSOR_NAME)
            embedding_tensor = sess.graph.get_tensor_by_name(
                vggish_params.OUTPUT_TENSOR_NAME)

            # Run inference and postprocessing.
            [embedding_batch] = sess.run([embedding_tensor],
                                         feed_dict={features_tensor: examples_batch})
            emb = []
            for embedding in pproc.postprocess(embedding_batch):
                emb.extend(embedding.tolist())
        label_tsv.append([ontology_lookup[label]['name']])
        audio_tsv.append([f'{label}/{filename}'])
        emb_tsv.append(emb)
        assert len(emb_tsv[0]) == len(emb)


        with open(f'{OUTPUTDIR}/emb.tsv', 'w') as f:
            for emb in emb_tsv:
                csv.writer(f, delimiter='\t').writerow(emb)
        with open(f'{OUTPUTDIR}/label.tsv', 'w') as f:
            for label in label_tsv:
                csv.writer(f, delimiter='\t').writerow(label)
        with open(f'{OUTPUTDIR}/audio.tsv', 'w') as f:
            for audio_path in audio_tsv:
                csv.writer(f, delimiter='\t').writerow(audio_path)


if __name__ == '__main__':
  tf.app.run()