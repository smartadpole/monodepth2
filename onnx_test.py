from torchvision.transforms import transforms

from onnxmodel import ONNXModel
from PIL import Image
import numpy as np
import cv2

net = ONNXModel("./monodepth2_stereo_640x192.onnx")

img_path = "/home/ljx/Code/200sever/work/sunhao/monodepth2/assets/test_image.jpg"

img = cv2.imread(img_path)
img = cv2.resize(img,(640,192),cv2.INTER_LANCZOS4)
img=img.transpose(2, 0, 1)
img = np.expand_dims(img,axis=0).astype("float32")/255.0

# origin_image = Image.open(img_path).convert('RGB')
# original_width, original_height = origin_image.size
# input_image = origin_image.resize((640, 192), Image.LANCZOS)
# input_image = transforms.ToTensor()(input_image).unsqueeze(0)
# img=input_image.cpu().numpy()

output = net.forward(img)
dis_array = output[0][0][0]
dis_array = (dis_array - dis_array.min()) / (dis_array.max() - dis_array.min()) * 255.0
dis_array = dis_array.astype("uint8")

showImg = cv2.resize(dis_array, (dis_array.shape[-1], dis_array.shape[0]))
showImg = cv2.applyColorMap(cv2.convertScaleAbs(showImg, 1), cv2.COLORMAP_PARULA)
cv2.imwrite("onnx_result.jpg", showImg)