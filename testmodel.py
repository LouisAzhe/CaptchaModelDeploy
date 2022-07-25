import torch
import Captcha_Model

device = torch.device("cuda:0")
use_gpu = torch.cuda.is_available()

model = torch.load('densenet121_ep50_fulldata.pkl')
if use_gpu == True:
    model = model.cuda(device)
rdl1 = Captcha_Model.return_dataloader('image.jpg')
result = Captcha_Model.test_model(model, rdl1)
print(result)