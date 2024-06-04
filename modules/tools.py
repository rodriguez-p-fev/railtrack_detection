import sys
import torch
import torch.nn as nn
from torch_snippets import *
import cv2
import numpy as np
import time
import os
from PIL import Image

bce = nn.BCEWithLogitsLoss()
def BCELoss(preds, targets):
	ce_loss = bce(preds, targets)
	acc = (torch.max(preds, 1)[1] == targets).float().mean()
	return ce_loss,  acc
def get_data_split(trainset_path, valset_path):
	f = open(trainset_path, "r")
	X_train = []
	for x in f:
		X_train.append((x+'.jpg').replace('\n',''))
	f.close()
	f = open(valset_path, "r")
	X_val = []
	for x in f:
		X_val.append((x+'.jpg').replace('\n',''))
	f.close()
	return X_train, X_val
def label_img(img, model, img_shape, DEVICE):
    model.eval()
    image = img.to(device=DEVICE)
    image = image[None]
    _mask = model.pred(image)
    _mask = torch.sigmoid(_mask)
    _mask = _mask.squeeze(1).permute(1,2,0).to("cpu")
    _mask = _mask.detach().numpy()
    _mask[_mask >= 0.5] = 1.0
    _mask[_mask < 0.5] = 0.0
    _mask = cv2.resize(_mask, (img_shape[1],img_shape[0]))
    return _mask
def blend_mask(img, pred):
    output = np.copy(img)
    idxs = np.where(pred == 1.0)
    for i in range(len(idxs[0])):
        output[idxs[0][i]][idxs[1][i]] = [255,0,228]
    return output
def create_masks(img_dl, model, cfg, device):
	log = Report(1)
	N = len(img_dl)
	for bx, data in enumerate(img_dl):
		images, names = data
		pred = label_img(images.__getitem__(0), model, (cfg['IMAGE_HEIGHT'],cfg['IMAGE_WIDTH']), device)
		for i in range(len(names)):
			img_path = os.path.join(cfg['WABTEC_VIDEO_IMAGES'], names[i])
			original_img = np.array(Image.open(img_path).convert("RGB"))
			resized = cv2.resize(original_img, (cfg['IMAGE_WIDTH'],cfg['IMAGE_HEIGHT']), interpolation = cv2.INTER_AREA)
			blended = blend_mask(resized, pred)
			mask = Image.fromarray(blended)
			mask.save(cfg['LABELED_DIR'] + names[i])
			log.record((bx+1)/N, end='\r')
	return None
def make_video(cfg):
	FRAME_RATE = 10
	archivos = sorted(os.listdir(cfg['LABELED_DIR']))
	img_array = []
	for x in range (0,len(archivos)):
		nomArchivo = archivos[x]
		dirArchivo = os.path.join(cfg['LABELED_DIR'], str(nomArchivo))
		img = cv2.imread(dirArchivo)
		resized = cv2.resize(img, (cfg['IMAGE_WIDTH'],cfg['IMAGE_HEIGHT']), interpolation = cv2.INTER_AREA)
		img_array.append(resized)
	video = cv2.VideoWriter(os.path.join(cfg['VIDEO_PATH'],cfg['VIDEO_NAME']), cv2.VideoWriter_fourcc(*'mp4v'), FRAME_RATE, (cfg['IMAGE_WIDTH'],cfg['IMAGE_HEIGHT']))
	for i in range(0, len(img_array)):
		video.write(img_array[i])
	video.release()
	return None
def save_summary(train_loss, train_acc, val_loss, val_acc, path):
	f = open(path, "a")
	string = f'{train_loss},{train_acc},{val_loss},{val_acc}\n'
	f.write(string)
	f.close()
	return None
def get_args():
	if len(sys.argv) == 2:
		action = sys.argv[1]
		by = 'unet'
	elif len(sys.argv) == 3:
		action = sys.argv[1]
		by = sys.argv[2]
	else:
		action = 'train'
		by = 'unet'
	if(action not in ['train','makevideo']):
		return None
	if(by not in ['unet','bisenet']):
		return None
	return [action,by]
