import numpy as np
import spacy

# Load the spacy model that you have installed
nlp = spacy.load('en_core_web_md')
# process a sentence using the model
doc = nlp("bark speech car guitar thunderstorm meow cymbal howl")
# It's that simple - all of the vectors and words are assigned after this point
# Get the vector for 'text':

# array = np.asarray([word.vector for word in doc])
# print(array.shape)
# mean = np.mean(array, axis=0)
# print(mean.shape)

for w1 in doc[1:]:
    for w2 in doc[1:]:
        d1 = np.linalg.norm(doc[0].vector - w1.vector)
        d2 = np.linalg.norm(doc[0].vector - w2.vector)

        if d1 < d2:
            print(f'{w1} -- {doc[0]} ------ {w2} (d1 < d2 | {d1} < {d2})')
        elif d1 > d2:
            print(f'{w1} ------ {doc[0]} -- {w2} (d1 > d2 | {d1} > {d2})')
        elif d1 == d2:
            print(f'{w1} -- {doc[0]} -- {w2} (d1 = d2 | {d1} = {d2})')