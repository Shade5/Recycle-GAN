import torch
from data.data_loader import CreateDataLoader
from options.test_options import TestOptions
from models.recycle_gan_model import RecycleGANModel
from tensorboardX import SummaryWriter

opt = TestOptions().parse()
opt.nThreads = 1  # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

writer = SummaryWriter('tbx/test_' + opt.name)
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = RecycleGANModel()
model.initialize(opt)

model.netG_A.eval()
model.netG_B.eval()

real_A_vid = []
fake_B_vid = []
for i, data in enumerate(dataset):
	if i >= 200:
		break
	fakeB = model.netG_A(data['A'].cuda())
	real_A_vid.append(data['A'][0]*0.5 + 0.5)
	fake_B_vid.append(fakeB[0].cpu().detach()*0.5 + 0.5)

	writer.add_image('A/real_A', real_A_vid[-1], i)
	writer.add_image('A/fake_b', fake_B_vid[-1], i)

writer.add_video('A/real_A', torch.stack(real_A_vid, dim=0).unsqueeze(0), 0)
writer.add_video('A/fake_B', torch.stack(fake_B_vid, dim=0).unsqueeze(0), 0)
