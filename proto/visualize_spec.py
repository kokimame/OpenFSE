import glob
import numpy as np
import matplotlib.pyplot as plt

SPECPATH = '/home/kokimame/Project/Master_Files/spec_tagged/_m_0b128/*.npy'

for file in glob.glob(SPECPATH):
    spec = np.load(file)
    plt.imshow(spec)
    plt.show()
