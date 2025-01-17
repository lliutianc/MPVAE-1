import os
import numpy as np


def build_path(*paths):
	for path in paths:
		path_levels = path.split('/')
		cur_path = ""
		for path_seg in path_levels:
			if len(cur_path):
				cur_path = cur_path + "/" + path_seg
			else:
				cur_path = path_seg
			if not os.path.exists(cur_path):
				os.mkdir(cur_path)


def get_label(data, order, offset, label_dim):
	output = []
	for i in order:
		output.append(data[i][offset:offset+label_dim])
	output = np.array(output, dtype="int") 
	return output


def get_feat(data, order, offset, label_dim, feature_dim):
	output = []
	offset = offset + label_dim
	for i in order:
		output.append(data[i][offset:offset + feature_dim])
	output = np.array(output, dtype="float32") 
	return output


def allexists(*files):
	exist = [os.path.exists(f) for f in files]
	return min(exist)


def search_files(path, prefix=None, postfix=None):
	files = []
	for file in os.listdir(path):
		if prefix:
			match_pre = True if file[:len(prefix)] == prefix else False
		else:
			match_pre = True
		if postfix:
			match_post = True if file[-len(postfix):] == postfix else False
		else:
			match_post = True
		if match_pre and match_post:
			files.append(file)
			
	return files
