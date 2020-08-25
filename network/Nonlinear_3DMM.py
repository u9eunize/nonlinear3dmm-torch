'''
import torch
from torch import nn

class Nonlinear3DMM(nn.Module):
	def __init__(self, config):
		...config...
		pass
		
	def forward(self, inputs):
		return None


class Helper():
	def __init__(self, config):
		...config...
		self.model = Nonlinear3DMM(config)
		pass
	
	
	def train(self):
		pass
	
		
	def eval(self):
		pass
	
		
	def predict(self):
		return None
'''