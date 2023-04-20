import os
import json
import torch
import shutil
import datetime
import numpy as np
from torch import optim
from os.path import join
from pushover import Client
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from torch.cuda.amp import GradScaler
from utilities.update_fig import update_fig
from utilities.fit import train_model, eval_model
# just edit here when change the net, dataloader, and accuracy function
from nets.template_net import TemplateNet as Network
from utilities.metrics import calculate_acc_1 as acc_func
from datasets.loaders.siamese_template_loader import get_dataloader as get_loader

if __name__ == '__main__':
    debug_mode = False  # when turn on the debug mode, code will not create checkpoints (including create folders)
    print('the code will run in debug mode') if debug_mode else None
    # define pushover client
    client = Client(config_path='/home/zg34/Documents/app_configs/pushover.conf', profile='S22U')
    # use try-except, if the code experienced a crash, send a notification to mobile phone
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'  # run on gpu if possible
        print('the code will run on {}'.format(device))
        # region Hyperparameters. get hyperparameters from config file and put those parameters in the variable "args"
        config_path = r'/home/zg34/Desktop/SignatureVerification/configs/config.json'
        parser = ArgumentParser()
        with open(config_path, 'r') as f:
            d = json.load(f)
            for k in d:
                parser.add_argument('--{}'.format(k), type=type(d[k]), default=d[k])
        args = parser.parse_args()
        # print hyperparameters in the variable "args"
        for arg in vars(args):
            print(arg, ' = ', getattr(args, arg))
        # endregion Hyperparameters.
        # region Figure-Checkpoints. prepare folders and events for checkpoints and figure
        if not debug_mode:  # if the code is in debug mode, don't any generate log
            log_dir = join(args.log_dir, 'checkpoints_{}'.format(datetime.datetime.now().strftime('%y%m%d_%H%M%S')))
            os.makedirs(log_dir, exist_ok=True)
            shutil.copy(config_path, log_dir)
            print('create the folder: {}\nand copy the config file in it'.format(log_dir))
        else:
            log_dir = None
        # generate the var to save losses and accuracies
        train_history = np.array([[None], [None]])
        eval_history = np.array([[None], [None]])
        if args.show_fig:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            fig.canvas.manager.set_window_title("Training process")
            plt.pause(0.1)
            update_fig(fig, ax1, ax2, train_history, eval_history)
            plt.pause(0.1)
            plt.ion()
        else:
            fig, ax1, ax2 = None, None, None
        # endregion Figure-Checkpoints.
        # region Components. define necessary components and move some of them to device
        model = Network().to(device)
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        train_loader = get_loader(is_train=True, batch_size=args.batch_size, shuffle=True, debug_mode=debug_mode)
        test_loader = get_loader(is_train=False, batch_size=args.batch_size, shuffle=False, debug_mode=debug_mode)
        # endregion Components.
        scaler = GradScaler() if args.fp16 else None  # if the code uses FP16 to speed up the training process
        for epoch in range(1, args.num_epochs + 1):  # Start training
            # region Main-flow. main flow for training a neural network
            train_loss, train_acc = train_model(model, optimizer, criterion, train_loader, [epoch, args.num_epochs],
                                                device, acc_func, scaler)
            train_history = np.concatenate((train_history, np.array([[train_loss], [train_acc]])), axis=1)
            eval_loss, eval_acc = eval_model(model, criterion, test_loader, [epoch, args.num_epochs], device, acc_func,
                                             scaler)
            eval_history = np.concatenate((eval_history, np.array([[eval_loss], [eval_acc]])), axis=1)
            # endregion Main-flow.
            # region Miscellaneous. execute update figure, push notifications, and save checkpoints
            if args.show_fig:
                update_fig(fig, ax1, ax2, train_history, eval_history)
                plt.pause(0.1)
            # if code is in debug_mode, don't push notifications and save checkpoints
            if not debug_mode:  # when "debug_mode" is True, "not debug_mode" is False, so don't execute the following
                # push notifications
                if epoch % args.notif_interval == 0 or epoch == 1:
                    msg = 'Epoch {}/{}, train_loss={:.3f}, train_acc={:.2%}, eval_loss={:.3f}, eval_acc={:.2%}'.format(
                        epoch, args.num_epochs, train_loss, train_acc, eval_loss, eval_acc)
                    client.send_message(msg)
                # save checkpoints
                if epoch % args.save_interval == 0 or epoch == args.num_epochs:
                    to_save = {'model': model.state_dict(), 'optim': optimizer.state_dict(), 'train_his': train_history,
                               'eval_his': eval_history}
                    torch.save(to_save,
                               join(log_dir, 'epoch_{}_loss_{:.3f}_acc_{:.3f}.pt'.format(epoch, eval_loss, eval_acc)))
                    print('Saved checkpoints')
            # endregion Miscellaneous.
        print('\nFinished training\n')
        if args.show_fig:
            plt.ioff()
            plt.show()
    except Exception as ex:  # show error
        client.send_message('Programme got a crash:\n{}'.format(ex))
        print('Programme got a crash:\n{}'.format(ex))
