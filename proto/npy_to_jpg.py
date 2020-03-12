import tensorflow as tf
import numpy as np
import datetime
import random
from PIL import Image

# def convert_and_save_image(image):
#     encoded_image = tf.image.encode_jpeg(image, format="grayscale")
#     writeOp = tf.write_file(datetime.datetime.now().strftime("%Y%m%d%H%M%S")
#                             + str(round(random.random() * 1000)) + ".jpg",
#                             encoded_image)
#     return writeOp
#

# casted_image_tf = tf.cast(x_adv_sub, tf.uint8)
TESTFILE = "/home/kokimame/Project/Master_Files/spec/7128_2.npy"

spec = np.load(TESTFILE)
spec = spec.astype(np.uint8)

print(spec)
# spec = tf.expand_dims(spec, 1)
# tf.io.encode_jpeg(spec)
#
# encoded_images_tf = tf.map_fn(convert_and_save_image, spec)
# tf.Session.session.run(encoded_images_tf)

img = Image.fromarray(np.squeeze(spec)) # Pixels are in range 0 to 1 and need to be in 0-255 for PIL
#img.show()
path = "test_spec.jpeg"
img.save(path, "JPEG")