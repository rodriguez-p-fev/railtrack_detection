import torch
from .UNet import UNet
from .tools import BCELoss

def set_model(cfg):
    model = UNet().to(device=cfg['DEVICE'])
    if(cfg['LOAD_MODEL']):
        print("UNet model loading")
        state_dict = torch.load(cfg['SAVED_UNET_MODEL'])
        model.load_state_dict(state_dict)
        model.to(cfg['DEVICE'])
    model.train()
    criteria = BCELoss
    return model, criteria
def set_optimizer(model, cfg):
	optim = torch.optim.Adam(
		model.parameters(),
		lr=cfg['LEARNING_RATE'],
		weight_decay=cfg['WEIGHT_DECAY'],
	)
	return optim
def train_batch(model, data, optim, criteria, device):
    model.train()
    im, lb = data
    im = im.to(device=device)
    lb = lb.float().unsqueeze(1).to(device=device)
    logits = model(im)
    optim.zero_grad()
    loss, acc = criteria(logits, lb)
    loss.backward()
    optim.step()
    return loss, acc

@torch.no_grad()
def validate_batch(model, data, criterion, device):
    model.eval()
    ims, targets = data
    images = ims.to(device=device)
    masks = targets.float().unsqueeze(1).to(device=device)
    _masks = model(images)
    loss, acc = criterion(_masks, masks)
    return loss, acc