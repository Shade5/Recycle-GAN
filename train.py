import time
from tensorboardX import SummaryWriter
from data.data_loader import CreateDataLoader
from models.models import create_model
from options.train_options import TrainOptions
from util.visualizer import Visualizer

opt = TrainOptions().parse()
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = create_model(opt)
visualizer = Visualizer(opt)
total_steps = 0
writer = SummaryWriter('tbx/test7')

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
	epoch_start_time = time.time()
	epoch_iter = 0

	for i, data in enumerate(dataset):
		iter_start_time = time.time()
		visualizer.reset()
		total_steps += opt.batchSize
		epoch_iter += opt.batchSize
		model.set_input(data)
		model.optimize_parameters()

		if total_steps % opt.display_freq == 0:
			save_result = total_steps % opt.update_html_freq == 0
			v = model.get_current_visuals()
			writer.add_image('A/real_A0', v['real_A0'], epoch)
			writer.add_image('A/real_A1', v['real_A1'], epoch)
			writer.add_image('A/real_A2', v['real_A2'], epoch)
			writer.add_image('A/fake_B0', v['fake_B0'], epoch)
			writer.add_image('A/fake_B1', v['fake_B1'], epoch)
			writer.add_image('A/fake_B2', v['fake_B2'], epoch)

			writer.add_image('B/real_B0', v['real_B0'], epoch)
			writer.add_image('B/real_B1', v['real_B1'], epoch)
			writer.add_image('B/real_B2', v['real_B2'], epoch)
			writer.add_image('B/fake_A0', v['fake_A0'], epoch)
			writer.add_image('B/fake_A1', v['fake_A1'], epoch)
			writer.add_image('B/fake_A2', v['fake_A2'], epoch)

		if total_steps % opt.print_freq == 0:
			errors = model.get_current_errors()
			t = (time.time() - iter_start_time) / opt.batchSize
			visualizer.print_current_errors(epoch, epoch_iter, errors, t)
			if opt.display_id > 0:
				visualizer.plot_current_errors(epoch, float(epoch_iter) / dataset_size, opt, errors)

		if total_steps % opt.save_latest_freq == 0:
			print('saving the latest model (epoch %d, total_steps %d)' %
				  (epoch, total_steps))
			model.save('latest')


	print('End of epoch %d / %d \t Time Taken: %d sec' %
		  (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
	model.update_learning_rate()
