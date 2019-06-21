from data.data_loader import CreateDataLoader
from options.test_options import TestOptions
from models.recycle_gan_model import RecycleGANModel
from tensorboardX import SummaryWriter
from util.util import tensor2im
import torch

opt = TestOptions().parse()
opt.nThreads = 1  # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = RecycleGANModel()
model.initialize(opt)
model.netG_A.eval()
writer = SummaryWriter('tbx/videotest')
# create website
# test
real_A_vid = []
fake_B_vid = []
for i, data in enumerate(dataset):
	if i == 3:
		break
	fakeB = model.netG_A(data['A'].cuda())
	real_A_vid.append(tensor2im(data['A']))
	fake_B_vid.append(tensor2im(fakeB))

writer.add_video('A/real_A', torch.stack(real_A_vid, dim=0).unsqueeze(0).permute(0, 2, 1, 3, 4), 0)
writer.add_video('A/fake_B', torch.stack(fake_B_vid, dim=0).unsqueeze(0).permute(0, 2, 1, 3, 4), 0)

