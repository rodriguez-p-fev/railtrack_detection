import sys
from config import cfg
import torch.cuda.amp as amp
from torch_snippets import *
import modules.tools as tools
import modules.data as dataloaders
import modules.data_tools as datatools
import modules.unet_tools as unet
import modules.bisenet_tools as bisenet
from modules.UNet import UNet
from modules.BiSeNetv2 import BiSeNetV2

def main():
	args = tools.get_args()
	if(args == None):
		print("Not valid arguments\nFirst argument is action\nSecond argument is model name\nValid actions:[train, makevideo]\nValid models:[unet, bisenet]\nTry again...")
	else:
		val_img_path = "/home/robothuman/Documents/data_sets/railways_datasets/ErieTestTrack_20231109/10Hzframes"
		val_img = "frame0-00-02.32.jpg"
		img = np.array(Image.open(os.path.join(val_img_path,val_img)).convert("RGB"))
		if(args[0] == 'train'):
			print(f'start {args[1]} training')
			if(args[1] == 'unet'):
				MEAN, STD = datatools.get_data_metrics(cfg['DATA.cfg'])
				train_transform, val_transform = datatools.get_train_transforms(cfg['DATA.cfg'], MEAN, STD)
				X_train, X_val = datatools.get_data_split(cfg['DATA.cfg']["TRAIN_LISTS_PATH"], cfg['DATA.cfg']["VALID_LISTS_PATH"])
				trn_dl, val_dl = datatools.get_loaders(X_train, X_val, train_transform, val_transform, cfg['DATA.cfg'])

				model, criterion = unet.set_model(cfg['MODEL.cfg'])
				optimizer = unet.set_optimizer(model,cfg['MODEL.cfg'])

				log = Report(cfg['SESSION.cfg']["NUM_EPOCHS"])
				for ex in range(cfg['SESSION.cfg']["NUM_EPOCHS"]):
					N = len(trn_dl)
					train_epoch_losses, train_epoch_accuracies = [], []
					for bx, data in enumerate(trn_dl):
						loss, acc = unet.train_batch(model, data, optimizer, criterion, cfg['MODEL.cfg']['DEVICE'])
						log.record(ex+(bx+1)/N,trn_loss=loss, trn_acc=acc, end='\r')
						train_epoch_losses.append(loss.item())
						train_epoch_accuracies.append(acc.item())
					
					N = len(val_dl)
					val_epoch_losses, val_epoch_accuracies = [], []
					for bx, data in enumerate(val_dl):
						loss, acc = unet.validate_batch(model, data, criterion, cfg['MODEL.cfg']['DEVICE'])
						log.record(ex+(bx+1)/N,val_loss=loss,val_acc=acc, end='\r')
						val_epoch_losses.append(loss.item())
						val_epoch_accuracies.append(acc.item())

					train_loss = np.array(train_epoch_losses).mean()
					MODEL = os.path.join(os.getcwd(),f"models/UNet_log/UNet_session{cfg['SESSION.cfg']['SESSION']}_epoch{ex}_v{cfg['DATA.cfg']['DATA_VERSION']}_loss{round(train_loss,3)}.pth")
					torch.save(model.to('cpu').state_dict(), MODEL)
					state_dict = torch.load(MODEL)
					model.load_state_dict(state_dict)
					model.to(cfg['MODEL.cfg']['DEVICE'])
					log.report_avgs(ex+1)
			elif(args[1] == 'bisenet'):
				cfg['DATA.cfg']['IMAGE_HEIGHT'] = cfg['DATA.cfg']['BISENET_HEIGHT']
				cfg['DATA.cfg']['IMAGE_WIDTH'] = cfg['DATA.cfg']['BISENET_WIDTH']
				MEAN, STD = datatools.get_data_metrics(cfg['DATA.cfg'])
				train_transform, val_transform = datatools.get_train_transforms(cfg['DATA.cfg'], MEAN, STD)
				X_train, X_val = datatools.get_data_split(cfg['DATA.cfg']["TRAIN_LISTS_PATH"], cfg['DATA.cfg']["VALID_LISTS_PATH"])
				trn_dl, val_dl = datatools.get_loaders(X_train, X_val, train_transform, val_transform, cfg['DATA.cfg'])

				model, criteria_pre, criteria_aux = bisenet.set_model(cfg['MODEL.cfg'])
				optimizer = bisenet.set_optimizer(model,cfg['MODEL.cfg'])
				scaler = amp.GradScaler()

				log = Report(cfg['SESSION.cfg']["NUM_EPOCHS"])
				for ex in range(cfg['SESSION.cfg']["NUM_EPOCHS"]):
					N = len(trn_dl)
					train_epoch_losses = []
					for bx, data in enumerate(trn_dl):
						data_batch_size = data[0].size()[0]
						if(data_batch_size > 1):
							loss, acc = bisenet.train_batch(model, data, scaler, optimizer, criteria_pre, criteria_aux, cfg['MODEL.cfg']['DEVICE'])
							train_epoch_losses.append(loss.item())
							log.record(ex+(bx+1)/N,trn_loss=loss, end='\r')

					N = len(val_dl)
					val_epoch_losses, val_epoch_accuracies = [], []
					for bx, data in enumerate(val_dl):
						loss, acc = bisenet.validate_batch(model, data, criteria_pre, cfg['MODEL.cfg']['DEVICE'])
						log.record(ex+(bx+1)/N,val_loss=loss,val_acc=acc, end='\r')
						val_epoch_losses.append(loss.item())
						val_epoch_accuracies.append(acc.item())

					train_loss = np.array(train_epoch_losses).mean()
					MODEL = os.path.join(os.getcwd(),f"models/BiSeNet_log/BiSeNet_session{cfg['SESSION.cfg']['SESSION']}_epoch{ex}_v{cfg['DATA.cfg']['DATA_VERSION']}_loss{round(train_loss,3)}.pth")
					torch.save(model.to('cpu').state_dict(), MODEL)
					state_dict = torch.load(MODEL)
					model.load_state_dict(state_dict)
					model.to(cfg['MODEL.cfg']['DEVICE'])
					log.report_avgs(ex+1)

					val_transform = datatools.get_label_transforms(cfg['DATA.cfg'], MEAN, STD)
					val_data = dataloaders.TestDataset([val_img], val_img_path, val_transform)
					test_img, name = val_data.__getitem__(0)
					mask = tools.label_img(test_img, model, img.shape, cfg['MODEL.cfg']['DEVICE'])
					blended = tools.blend_mask(img, mask)
					img_rec = Image.fromarray(blended)
					img_rec.save(f"/home/robothuman/Documents/data_sets/outputs/results/img_epoch_{ex}.png")
		elif(args[0] == 'makevideo'):
			print(f'making video with {args[1]}')
			if(args[1] == 'unet'):
				model = UNet().to(device=cfg['MODEL.cfg']['DEVICE'])
				state_dict = torch.load(cfg['MODEL.cfg']['SAVED_UNET_MODEL'])
				cfg['DATA.cfg']['VIDEO_NAME'] = f"{cfg['MODEL.cfg']['UNET_MODEL'].split('.')[0]}.mp4"
			elif(args[1] == 'bisenet'):
				cfg['DATA.cfg']['IMAGE_HEIGHT'] = cfg['DATA.cfg']['BISENET_HEIGHT']
				cfg['DATA.cfg']['IMAGE_WIDTH'] = cfg['DATA.cfg']['BISENET_WIDTH']
				model = BiSeNetV2().to(device=cfg['MODEL.cfg']['DEVICE'])
				state_dict = torch.load(cfg['MODEL.cfg']['SAVED_BISENET_MODEL'])
				cfg['DATA.cfg']['VIDEO_NAME'] = f"{cfg['MODEL.cfg']['BISENET_MODEL'].split('.')[0]}.mp4"
			
			model.load_state_dict(state_dict)
			model.to(cfg['MODEL.cfg']['DEVICE'])
			MEAN, STD = datatools.get_data_metrics(cfg['DATA.cfg'])
			transform = datatools.get_label_transforms(cfg['DATA.cfg'], MEAN, STD)
			images =  sorted(os.listdir(cfg['DATA.cfg']['WABTEC_VIDEO_IMAGES']))
			img_dl = dataloaders.get_test_loaders(images, cfg['DATA.cfg']['WABTEC_VIDEO_IMAGES'], transform)
			tools.create_masks(img_dl, model, cfg['DATA.cfg'], cfg['MODEL.cfg']['DEVICE'])
			tools.make_video(cfg['DATA.cfg'])

if __name__ == "__main__":
	main()
