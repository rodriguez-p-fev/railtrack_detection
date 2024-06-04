from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from .data import get_measure_loaders, get_loaders

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
def get_items(trainset_path):
	f = open(trainset_path, "r")
	X_train = []
	for x in f:
		X_train.append((x+'.jpg').replace('\n',''))
	f.close()
	return X_train
def get_mean_std(loader):
	num_pixels = 0
	num_images = 1
	for bx, data in enumerate(loader):
		if(bx == 0):
			images = data
			img_mean, img_std = images.mean([2,3]), images.std([2,3])
			batch_size, num_channels, height, width = images.shape
			num_pixels += height * width
			mean = img_mean
			std = img_std
		else:
			images = data
			img_mean, img_std = images.mean([2,3]), images.std([2,3])
			batch_size, num_channels, height, width = images.shape
			num_pixels += height * width
			mean += img_mean
			std += img_std
			num_images += 1
	mean = mean[0]
	std = std[0]
	for i in range(3):
		mean[i] = mean[i]/num_images
		std[i] = std[i]/num_images
	return mean, std
def get_data_metrics(data_cfg):
	if(data_cfg['GET_DATA_METRICS']):
		transform = transforms.Compose([
    		transforms.ToTensor(),
		])
		X_train = get_items(data_cfg["TRAIN_LISTS_PATH"])
		dl = get_measure_loaders(X_train, data_cfg["IMG_DIR"], transform)
		MEAN, STD = get_mean_std(dl)
	else:
		MEAN = [0.0, 0.0, 0.0]
		STD=[1.0, 1.0, 1.0]
	return MEAN, STD
def get_train_transforms(cfg,mean,std):
	train_transform = A.Compose([
        #A.Crop(x_min=int(cfg["ORIGINAL_WIDTH"]*0.3),y_min=300,x_max=int(cfg["ORIGINAL_WIDTH"]*0.7),y_max=cfg["ORIGINAL_HEIGHT"]-300,always_apply=True,p=0.7),
        A.Resize(height=cfg["IMAGE_HEIGHT"], width=cfg["IMAGE_WIDTH"]),
        A.HorizontalFlip(p=0.5),
        A.Rotate (limit=40, interpolation=1, border_mode=4, p=0.5),
        #A.RandomBrightnessContrast(brightness_limit=[-0.2,0.2], contrast_limit=[-0.3,0.3], p=0.5),
        #A.Blur(blur_limit=4, p=0.5),
        #A.ColorJitter(brightness=[1.0,1.0], contrast=[1.0,1.0], saturation=[0.5,2.0], hue=0.2, p=0.5),
        A.Normalize(
            mean=mean,
            std=std,
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
	])
	val_transform = A.Compose([
        A.Resize(height=cfg["IMAGE_HEIGHT"], width=cfg["IMAGE_WIDTH"]),
        A.Normalize(
            mean=mean,
            std=std,
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])
	return train_transform, val_transform
def get_label_transforms(cfg,mean,std):
	transform = A.Compose([
        A.Resize(height=cfg["IMAGE_HEIGHT"], width=cfg["IMAGE_WIDTH"]),
        A.Normalize(
            mean=mean,
            std=std,
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])
	return transform