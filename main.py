import cv2
import numpy as np
import os

# Set the paths to the model files
prototxt_path = os.path.abspath('models\colorization_deploy_v2.prototxt')
model_path = os.path.abspath('models\colorization_release_v2.caffemodel')
kernel_path = os.path.abspath('models\pts_in_hull.npy')
image_path = os.path.abspath('images/scene.jpeg')

# Ensure the paths are correct
assert os.path.exists(prototxt_path), f"Prototxt file not found: {prototxt_path}"
assert os.path.exists(model_path), f"Model file not found: {model_path}"
assert os.path.exists(kernel_path), f"Kernel file not found: {kernel_path}"
assert os.path.exists(image_path), f"Image file not found: {image_path}"

# Load the model
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Load the cluster centers
points = np.load(kernel_path)
points = points.transpose().reshape(2, 313, 1, 1)
net.getLayer(net.getLayerId("class8_ab")).blobs = [points.astype(np.float32)]
net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# Load the input image
bw_image = cv2.imread(image_path)
if bw_image is None:
    raise FileNotFoundError(f"Failed to load image: {image_path}")

# Normalize the image
normalized = bw_image.astype("float32") / 255.0
lab = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)

# Resize the L channel to network's expected input size
resized = cv2.resize(lab, (224, 224))
L = cv2.split(resized)[0]
L -= 50

# Predict the a and b channels
net.setInput(cv2.dnn.blobFromImage(L))
ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

# Resize the predicted a and b channels to match the input image size
ab = cv2.resize(ab, (bw_image.shape[1], bw_image.shape[0]))
L = cv2.split(lab)[0]

# Combine with the original L channel
colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
colorized = (255.0 * colorized).astype("uint8")

# Display the images
cv2.imshow("BW Image", bw_image)
cv2.imshow("Colorized", colorized)
cv2.waitKey(0)
cv2.destroyAllWindows()
