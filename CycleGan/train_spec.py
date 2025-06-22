import time
import sys
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset
    dataset_size = len(dataset)    # number of training images
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create the model
    model.setup(opt)               # setup: load networks; create schedulers

    total_iters = 0                # total training iterations

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # loop over epochs
        epoch_start_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(dataset):  # loop within one epoch
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)           # unpack data
            model.optimize_parameters()     # optimize

            if total_iters % opt.print_freq == 0:    # print training losses
                losses = model.get_current_losses()
                loss_str = ' '.join([f'{k}: {v:.4f}' for k, v in losses.items()])
                print(f'(epoch: {epoch}, iters: {total_iters}) {loss_str}')

            if total_iters % opt.save_latest_freq == 0:   # save latest model
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

        if epoch % opt.save_epoch_freq == 0:              # save model at the end of epoch
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rate
