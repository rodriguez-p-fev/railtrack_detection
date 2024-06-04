import os
from os.path import isfile
from configparser import ConfigParser
from datetime import datetime
import torch

cfg = None

PROJECT_PATH = os.getcwd()

if not isfile(os.path.join(os.getcwd(), "config.ini")):
	print("config file doesn't exist")
else:
	config = ConfigParser()
	config.read(os.path.join(os.getcwd(), 'config.ini'))
	
	model_cfg = {
		"UNET_MODEL":          config.get('config','UNET_MODEL'),
		"BISENET_MODEL":       config.get('config','BISENET_MODEL'),
		"SAVED_UNET_MODEL":    os.path.join(os.getcwd(),f"models/UNet/{config.get('config','UNET_MODEL')}"),
		"SAVED_BISENET_MODEL": os.path.join(os.getcwd(),f"models/BiSeNet/{config.get('config','BISENET_MODEL')}"),
		"LOAD_MODEL":          config.getboolean("config","LOAD_MODEL"),
		"LEARNING_RATE":       config.getfloat("config","LEARNING_RATE"),
		"WEIGHT_DECAY":        config.getfloat("config","WEIGHT_DECAY"),
		"DEVICE":              "cuda" if torch.cuda.is_available() else "cpu",
	}
	
	data_cfg = {
		"DATA_VERSION":        config.get("config","DATA_VERSION"),
		"IMG_DIR":             os.path.join(config.get("config","DATASET_PATH"),f"{config.get('config','DATASET_NAME')}{config.get('config','RAILSEM_IMG_DIR')}"),
		"MASK_DIR":            os.path.join(config.get("config","DATASET_PATH"),f"{config.get('config','DATASET_NAME')}{config.get('config','RAILSEM_MASK_DIR')}"),
		"TRAIN_LISTS_PATH":    os.path.join(config.get("config","DATASET_PATH"),f"{config.get('config','DATASET_NAME')}{config.get('config','TRAIN_LISTS_PATH')}"),
		"VALID_LISTS_PATH":    os.path.join(config.get("config","DATASET_PATH"),f"{config.get('config','DATASET_NAME')}{config.get('config','VALID_LISTS_PATH')}"),
		"WABTEC_VIDEO_IMAGES": os.path.join(config.get("config","DATASET_PATH"),config.get("config","WABTEC_VIDEO_IMAGES")),
		"LABELED_DIR":         os.path.join(config.get("config","DATASET_PATH"),config.get("config","LABELED_DIR")),
		"VIDEO_PATH":          os.path.join(config.get("config","DATASET_PATH"),"outputs/videos/"),
		"GET_DATA_METRICS":    config.getboolean("config","GET_MEAN_STD"),
		"ORIGINAL_HEIGHT":     config.getint("config","ORIGINAL_HEIGHT"),
		"ORIGINAL_WIDTH":      config.getint("config","ORIGINAL_WIDTH"),
		"IMAGE_HEIGHT":        int(config.getfloat("config","SCALING")*config.getint("config","ORIGINAL_HEIGHT")),
		"IMAGE_WIDTH":         int(config.getfloat("config","SCALING")*config.getint("config","ORIGINAL_WIDTH")),
		"BISENET_HEIGHT":      config.getint("config","BISENET_HEIGHT"),
		"BISENET_WIDTH":       config.getint("config","BISENET_WIDTH"),
		"RAILS_CODE":          config.getint("config","RAILS_CODE"),
		"BATCH_SIZE":          config.getint("config","BATCH_SIZE"),
	}
	
	session_cfg = {
		"SESSION":             f"{datetime.now().month}-{datetime.now().day}-{datetime.now().hour}-{datetime.now().minute}",
		"NUM_EPOCHS":          config.getint("config","NUM_EPOCHS"),
	}
	
	cfg = {
		"MODEL.cfg":   model_cfg,
		"DATA.cfg":    data_cfg,
		"SESSION.cfg": session_cfg
	}
