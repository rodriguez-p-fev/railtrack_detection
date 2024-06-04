import torch
from .BiSeNetv2 import BiSeNetV2
from .tools import BCELoss

def set_model(cfg):
    model = BiSeNetV2().to(device=cfg['DEVICE'])
    if(cfg['LOAD_MODEL']):
        print("BiSeNet model loading")
        state_dict = torch.load(cfg['SAVED_BISENET_MODEL'])
        model.load_state_dict(state_dict)
        model.to(cfg['DEVICE'])
    model.train()
    criteria_pre = BCELoss
    criteria_aux = [BCELoss for _ in range(4)]
    return model, criteria_pre, criteria_aux
def set_optimizer(model, cfg):
	if hasattr(model, 'get_params'):
		wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = model.get_params()
		wd_val = cfg['WEIGHT_DECAY']
		params_list = [
			{'params': wd_params, },
            {'params': nowd_params, 'weight_decay': wd_val},
            {'params': lr_mul_wd_params, 'lr': cfg['LEARNING_RATE']},
            {'params': lr_mul_nowd_params, 'weight_decay': wd_val, 'lr': cfg['LEARNING_RATE']},
		]
	else:
		wd_params, non_wd_params = [], []
		for name, param in model.named_parameters():
			if param.dim() == 1:
				non_wd_params.append(param)
			elif param.dim() == 2 or param.dim() == 4:
				wd_params.append(param)
		params_list = [
			{'params': wd_params, },
			{'params': non_wd_params, 'weight_decay': cfg['WEIGHT_DECAY']},
		]
	optim = torch.optim.Adam(
		params_list,
		lr=cfg['LEARNING_RATE'],
		weight_decay=cfg['WEIGHT_DECAY'],
	)
	return optim
def train_batch(model, data, scaler, optim, criteria, criteria_aux, device):
	model.train()
	im, lb = data
	im = im.to(device=device)
	lb = lb.float().unsqueeze(1).to(device=device)
	if(device =='cpu'):
		logits, *logits_aux = model(im)
		optim.zero_grad()
		loss_pre, acc = criteria(logits, lb)
		loss_aux = [crit(lgt, lb)[0] for crit, lgt in zip(criteria_aux, logits_aux)]
		loss = loss_pre + sum(loss_aux)
		loss.backward()
		optim.step()
	else:
		with torch.cuda.amp.autocast(enabled=True):
			logits, *logits_aux = model(im)
			loss_pre, acc = criteria(logits, lb)
			loss_aux = [crit(lgt, lb)[0] for crit, lgt in zip(criteria_aux, logits_aux)]
			loss = loss_pre + sum(loss_aux)
		scaler.scale(loss).backward()
		scaler.step(optim)
		scaler.update()
		torch.cuda.synchronize()
	return loss, acc

@torch.no_grad()
def validate_batch(model, data, criterion, device):
    model.eval()
    im, lb = data
    im = im.to(device=device)
    lb = lb.float().unsqueeze(1).to(device=device)
    logits, *logits_aux = model(im)
    loss, acc = criterion(logits, lb)
    return loss, acc