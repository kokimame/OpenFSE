import glob
import numpy as np
import matplotlib.pyplot as plt

SPECPATH = '/home/kokimame/Project/Master_Files/spec_tagged_mcuts/_m_0dwtp/*.npy'

for file in glob.glob(SPECPATH):
    spec = np.load(file)
    plt.imshow(spec)
    plt.title(spec.shape)
    plt.show()
