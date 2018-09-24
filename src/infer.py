from net import SRNet
import torch
from torch.autograd import Variable
import argparse
import cv2
import os
import numpy as np
import scipy.misc
import shutil
import time

parser = argparse.ArgumentParser() ;
parser.add_argument('--model', type=str, help='model epoch number to load')
parser.add_argument('--image', type=bool, help='infer raw image',default=False)
parser.add_argument('--npy', type=bool, help='infer npy', default=False)
args = parser.parse_args() ;


if __name__ == '__main__':

	# loading model
	model = SRNet()#torch.load('../models/weights' + args.model + '.pt', map_location={'cuda:2': 'cpu'}) ;
	model = model.eval();
	if args.image:
   		file = os.listdir('test');	
		y = np.float32(cv2.imread('test/' + file[0])) / 255;
		y.shape = (1,3,32,100) 
		y = Variable(torch.from_numpy(y))
		_y = model(y)
		_y = _y.data.numpy() * 255;
		_y = np.uint8(_y)
		_y.shape = (32,100,3)
		print(_y.shape)
		cv2.imwrite('test_results/_y' + file[0], _y);
	elif args.npy:
		y = np.float32(np.load('../data/val.npy')).reshape(-1,3,32,100);
		gty = np.load('../data/val_labels.npy');
		print(y.shape, gty.shape);
		y = Variable(torch.from_numpy(y))
		preds = [];
		print('forward pass');
		
		batch_size = 256;	
		j = 0;
		
		while 1:
			by = y[j:j+batch_size];
			j = (j+batch_size)%y.shape[0];
			start_time = time.time();
			_y = model(by)
			print('Time taken for a batch', time.time() - start_time);
			_y = _y.data.numpy() * 255;
			_y = np.uint8(_y)
			preds += list(_y);
			if j < batch_size:
			 break; 
		
		_y = np.uint8(preds);
		print('_y shape', _y.shape);
		exit(0);
		if os.path.exists('../results'):
			shutil.rmtree('../results');
		os.makedirs('../results/_y');
		os.makedirs('../results/y');
		for i, img in enumerate(_y):
			scipy.misc.imsave('../results/_y/' + str(i) + '.png', img.reshape(32,100,3));
		for i, img in enumerate(gty):
			scipy.misc.imsave('../results/y/' + str(i) + '.png', img.reshape(32,100,3));
 


