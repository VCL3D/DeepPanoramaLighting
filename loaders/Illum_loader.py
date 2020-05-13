from skimage import io, transform
import numpy as np
import cv2
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms, utils
'''
	Input (256,512,3)
'''
class IlluminationModule(nn.Module):
	def __init__(self, batch_size):
		super().__init__()
		self.cv_block1 = conv_bn_elu(3, 64, kernel_size=7, stride=2) 
		self.cv_block2 = conv_bn_elu(64, 128, kernel_size=5, stride=2) 
		self.cv_block3 = conv_bn_elu(128, 256, stride=2) 
		self.cv_block4 = conv_bn_elu(256, 256)
		self.cv_block5 = conv_bn_elu(256, 256, stride=2)  
		self.cv_block6= conv_bn_elu(256, 256)
		self.cv_block7 = conv_bn_elu(256, 256, stride=2) 
		self.fc = nn.Linear(256*16*8, 2048)
		'''One head regression'''
		self.sh_fc = nn.Linear(2048, 27)

	def forward(self, x):
		x = self.cv_block1(x)
		x = self.cv_block2(x)
		x = self.cv_block3(x)
		x = self.cv_block4(x)
		x = self.cv_block5(x)
		x = self.cv_block6(x)
		x = self.cv_block7(x)
		x = x.view(-1, 256*8*16)
		x = F.elu(self.fc(x))
		return((self.sh_fc(x)))

def conv_bn_elu(in_, out_, kernel_size=3, stride=1, padding=True):
	# conv layer with ELU activation function 
	pad = int(kernel_size/2)
	if padding is False:
		pad = 0
	return nn.Sequential(
		nn.Conv2d(in_, out_, kernel_size, stride=stride, padding=pad),
		nn.ELU(),
	)
	
class Inference_Data(Dataset):
	def __init__(self, img_path):
		self.input_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
		self.input_img = cv2.resize(self.input_img, (512,256), interpolation=cv2.INTER_CUBIC)
		self.to_tensor = transforms.ToTensor()
		self.data_len = 1

	def __getitem__(self, index):
		self.tensor_img = self.to_tensor(self.input_img)
		return self.tensor_img

	def __len__(self):
		return self.data_len