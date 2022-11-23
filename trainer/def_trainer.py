import shutil
import random
import numpy as np
import torch
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from utils.metrics import niqe
from loss.gan_arch import define_D
from loss import GuidedPerceptualLoss
from model.modules.squeeze import squeeze2d
from torchvision import transforms


to_image = transforms.Compose([transforms.ToPILImage()])


class DefaultTrainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        self.config = config

        self.split_point = self._safe_read(["arch", "args", "split_point"], 2)
        self.niqe_threshold = self._safe_read(["trainer", "niqe_threshold"], 3.40)
        self.special_epoch = self._safe_read(["trainer", "special_epoch"], 0)

        # self.netD_low_ft = define_D(input_nc=3 * self.split_point, ndf=64, n_layers_D=5)
        # self.netD_high_ft = define_D(input_nc=3 * self.split_point, ndf=64, n_layers_D=5)
        self.netD_low = define_D(input_nc=3 * 2, ndf=64, n_layers_D=5)
        self.netD_high = define_D(input_nc=3 * 2, ndf=64, n_layers_D=5)

        self.optimizer_D_low = config.init_obj('optimizer_D', torch.optim, self.netD_low.parameters())
        self.optimizer_D_high = config.init_obj('optimizer_D', torch.optim, self.netD_high.parameters())
        # self.optimizer_D_low_ft = config.init_obj('optimizer_D', torch.optim, self.netD_low_ft.parameters())
        # self.optimizer_D_high_ft = config.init_obj('optimizer_D', torch.optim, self.netD_high_ft.parameters())
        
        super().__init__(model, criterion, metric_ftns, optimizer, config)

        self.criterion = self.criterion()
        self.vgg_perceptual_loss = GuidedPerceptualLoss(last_only=self._safe_read(["trainer", "last_only"], False))

        self.guided_vgg_weight = config["trainer"]["guided_vgg_weight"]
        self.device = device
        self.data_loader = data_loader

        self.lr_scheduler_G = lr_scheduler
        if self.lr_scheduler_G is not None:
            self.lr_scheduler_D_low = config.init_obj('lr_scheduler_D', torch.optim.lr_scheduler, self.optimizer_D_low)
            self.lr_scheduler_D_high = config.init_obj('lr_scheduler_D', torch.optim.lr_scheduler, self.optimizer_D_high)
        # self.lr_scheduler_D_low_ft = config.init_obj('lr_scheduler_D', torch.optim.lr_scheduler, self.optimizer_D_low_ft)
        # self.lr_scheduler_D_high_ft = config.init_obj('lr_scheduler_D', torch.optim.lr_scheduler, self.optimizer_D_high_ft)
        
        self.l1_loss = torch.nn.L1Loss()
        self.consistency_weight = config["trainer"]["consistency_weight"]
        self.IAF_weight = config["trainer"]["IAF_weight"]

        self.logger.warning("weight of guided unpaired vgg loss is:" + str(self.guided_vgg_weight))
        self.logger.warning("weight of cycle consistency loss is:" + str(self.consistency_weight))
        self.logger.warning("weight of IAF loss is:" + str(self.IAF_weight))
        self.logger.warning("split point is:" + str(self.split_point))
        self.logger.warning("threshold of niqe is:" + str(self.niqe_threshold))
        self.logger.warning("the special epoch is:" + str(self.special_epoch))

        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None and self.config["trainer"]["monitor"] != "off"
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('lossG', 'lossG_ft', 'loss_guided', 'loss_consistency',
                                           'lossD_low', 'lossD_high', 'lossD_low_ft', 'lossD_high_ft',
                                           'loss_IAF', writer=self.writer)
        self.valid_metrics = MetricTracker('niqe', writer=self.writer)

        self.fitting_low = self.config.log_dir / "fitting/low/"
        self.fitting_high = self.config.log_dir / "fitting/high/"
        self.fitting_low.mkdir(parents=True, exist_ok=True)
        self.fitting_high.mkdir(parents=True, exist_ok=True)
        self.uval = self.config.log_dir / "uval/"
        self.uval_last = None
        self.uval_best = None

    def _train_epoch(self, epoch):
        """
        Training procedure for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.netD_low.train()
        self.netD_high.train()
        # self.netD_low_ft.train()
        # self.netD_high_ft.train()
        self.train_metrics.reset()
        lossG_sum = 0
        lossG_ft_sum = 0
        lossD_low_sum = 0
        lossD_high_sum = 0
        lossD_low_ft_sum = 0
        lossD_high_ft_sum = 0
        loss_guided_sum = 0
        loss_consistency_sum = 0
        loss_IAF_sum = 0
        for batch_idx, (low_real, high_real) in enumerate(self.data_loader):
            low_real, high_real = low_real.to(self.device), high_real.to(self.device)

            low_fake, loss_IAF_deg = self.model(high_real, rev=True)
            high_fake, loss_IAF_enh = self.model(low_real, rev=False)
            loss_IAF = (loss_IAF_enh + loss_IAF_deg) * self.IAF_weight
            if self.config["n_gpu"] > 1:
                loss_IAF = torch.sum(loss_IAF)

            low_real_remake, _ = self.model(high_fake.clamp(0, 1), rev=True)
            high_real_remake, _ = self.model(low_fake.clamp(0, 1), rev=False)

            loss_consistency = (self.l1_loss(low_real_remake, low_real) + self.l1_loss(high_real_remake, high_real)) \
                               * self.consistency_weight

            # low_real, high_real, low_fake, high_fake = self._choose_channel_rand([low_real, high_real, low_fake, high_fake])
            ft_list, img_list = self._split([low_real, high_real, low_fake, high_fake])
            lossD_low, lossD_high = self._backward_D(*img_list, self.optimizer_D_low, self.optimizer_D_high,
                                                     self.netD_low, self.netD_high)
            lossD_low_ft, lossD_high_ft = self._backward_D(*ft_list, self.optimizer_D_low, self.optimizer_D_high,
                                                           self.netD_low, self.netD_high)

            lossG, lossG_ft, loss_guided, loss_consistency, loss_IAF = self._backward_G(low_real, high_real, low_fake,
                                                                                high_fake, loss_consistency, loss_IAF,
                                                                                ft_list, img_list)

            lossD_low_sum += lossD_low.item()
            lossD_high_sum += lossD_high.item()
            lossD_low_ft_sum += lossD_low_ft.item()
            lossD_high_ft_sum += lossD_high_ft.item()
            lossG_sum += lossG.item()
            lossG_ft_sum += lossG_ft.item()
            loss_guided_sum += loss_guided.item()
            loss_consistency_sum += loss_consistency.item()
            loss_IAF_sum += loss_IAF.item()

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} LossG: {:.6f} LossG_ft: {:.6f} \n\t'
                                  'lossD_low: {:.6f} lossD_high: {:.6f} '
                                  'lossD_low_ft: {:.6f} lossD_high_ft: {:.6f} \n\t'
                                  'Loss_guided: {:.6f} Loss_consistency: {:.6f} loss_IAF: {:.6f}\n\t'
                                  'lrG: {:.6f} lrD_low: {:.6f} lrD_high: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    lossG.item(),
                    lossG_ft.item(),
                    lossD_low.item(),
                    lossD_high.item(),
                    lossD_low_ft.item(),
                    lossD_high_ft.item(),
                    loss_guided.item(),
                    loss_consistency.item(),
                    loss_IAF.item(),
                    self.optimizer.param_groups[0]['lr'],
                    self.optimizer_D_low.param_groups[0]['lr'],
                    self.optimizer_D_high.param_groups[0]['lr']
                ))
                # self.writer.add_image('input', make_grid(low_real.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break

        self.writer.set_step(epoch)
        self.train_metrics.update('lossG', lossG_sum / len(self.data_loader))
        self.train_metrics.update('lossG_ft', lossG_ft_sum / len(self.data_loader))
        self.train_metrics.update('loss_guided', loss_guided_sum / len(self.data_loader))
        self.train_metrics.update('loss_consistency', loss_consistency_sum / len(self.data_loader))
        self.train_metrics.update('lossD_low', lossD_low_sum / len(self.data_loader))
        self.train_metrics.update('lossD_high', lossD_high_sum / len(self.data_loader))
        self.train_metrics.update('lossD_low_ft', lossD_low_ft_sum / len(self.data_loader))
        self.train_metrics.update('lossD_high_ft', lossD_high_ft_sum / len(self.data_loader))
        self.train_metrics.update('loss_IAF', loss_IAF_sum / len(self.data_loader))
        log = self.train_metrics.result()

        if self.lr_scheduler_G is not None:
            self.lr_scheduler_G.step()
            self.lr_scheduler_D_low.step()
            self.lr_scheduler_D_high.step()
        # self.lr_scheduler_D_low_ft.step()
        # self.lr_scheduler_D_high_ft.step()

        with torch.no_grad():
            test_low, test_high = self.data_loader.dataset.ret_visuals()
            fitting_low = to_image(torch.clamp(self.model(x=test_high.to(self.device), rev=True)[0][0], 0, 1).squeeze().cpu())
            fitting_high = torch.clamp(self.model(x=test_low.to(self.device))[0][0], 0, 1)
            fitting_high = to_image(fitting_high.squeeze().cpu())
            fitting_low.save(str(self.fitting_low / ("%d.png" % epoch)))
            fitting_high.save(str(self.fitting_high / ("%d.png" % epoch)))

        if self.do_validation and epoch >= 100:
            val_log = self._valid_epoch(epoch)
            log.update(**{k :v for k, v in val_log.items()})


        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()

        if self.uval_last is not None:
            shutil.rmtree(str(self.uval_last))
        self.uval_last = self.uval / ("last(" + str(epoch) + ")")
        self._mkdir_val(self.uval_last, self.valid_data_loader.dataset.dataset_dir_list)
        with torch.no_grad():
            for batch_idx, (input_img, name) in enumerate(self.valid_data_loader):
                name = name[0]
                input_img = input_img.to(self.device)
                input_img = input_img[:, :, :(input_img.shape[2] // 2) * 2, :(input_img.shape[3] // 2) * 2]

                generated_img = torch.clamp(self.model(x=input_img, test_scale_shift=True)[0], 0, 1)
                generated_img = to_image(torch.squeeze(generated_img.float().detach().cpu()))
                generated_img.save(self.uval_last / name)
                # print("write " + name + " successful!")

        niqe_value = niqe(self.uval_last)

        if epoch == self.special_epoch:
            self.uval_spec = self.uval / str(self.special_epoch)
            shutil.copytree(str(self.uval_last), str(self.uval_spec))

        if niqe_value < self.niqe_threshold:
            uval_good = self.uval / ("epoch" + str(epoch) + ", niqe" + str(niqe_value))
            shutil.copytree(str(self.uval_last), str(uval_good))

        self.writer.set_step(epoch, 'valid')
        self.valid_metrics.update('niqe', niqe_value)
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _mkdir_val(self, base_dir, dataset_list=("DICM", "LIME", "NPE", "MEF", "VV", "RetinexNet/LOLdataset/eval15/low")):
        for dataset_name in dataset_list:
            dataset_dir = base_dir / dataset_name
            dataset_dir.mkdir(parents=True, exist_ok=True)

    def _choose_channel(self, batch_list, channel_idx):
        for i in range(len(batch_list)):
            batch = squeeze2d(batch_list[i])
            batch_list[i] = torch.cat((torch.unsqueeze(batch[:, channel_idx, :, :], dim=1),
                                       torch.unsqueeze(batch[:, channel_idx+4, :, :], dim=1),
                                       torch.unsqueeze(batch[:, channel_idx+8, :, :], dim=1)), dim=1)
    
        return batch_list
    
    def _split(self, tensor_list):
        feature_list = []
        image_list = []

        for tensor in tensor_list:
            tensor = squeeze2d(tensor)
            x_a_list = []
            x_b_list = []
            for i in range(3):
                channel_a = tensor[:, i * 4:i * 4 + self.split_point, :, :]
                channel_b = tensor[:, i * 4 + self.split_point:(i + 1) * 4, :, :]
                if len(channel_a.shape) == 3:
                    channel_a = torch.unsqueeze(channel_a, dim=1)
                if len(channel_b.shape) == 3:
                    channel_b = torch.unsqueeze(channel_b, dim=1)
                x_a_list.append(channel_a)
                x_b_list.append(channel_b)

            feature_list.append(torch.cat(x_a_list, 1))
            image_list.append(torch.cat(x_b_list, 1))

        return feature_list, image_list

    def _backward_D(self, low_real, high_real, low_fake, high_fake, optim_low, optim_high, net_low, net_high):
        optim_low.zero_grad()
        optim_high.zero_grad()

        # Train Discriminator with all real batch
        label = True
        judge_low_real = net_low(low_real).view(-1)  # low_light part
        lossD_low_real = self.criterion(judge_low_real, label)

        judge_high_real = net_high(high_real).view(-1)  # high_light part
        lossD_high_real = self.criterion(judge_high_real, label)

        # Train discriminator with all fake batch
        label = False
        judge_low_fake = net_low(low_fake.detach()).view(-1)  # low_light part
        lossD_low_fake = self.criterion(judge_low_fake, label)

        judge_high_fake = net_high(high_fake.detach()).view(-1)  # high_light part
        lossD_high_fake = self.criterion(judge_high_fake, label)

        # Discriminator step
        lossD_low = lossD_low_fake + lossD_low_real
        lossD_low.backward()
        optim_low.step()

        lossD_high = lossD_high_fake + lossD_high_real
        lossD_high.backward()
        optim_high.step()

        return lossD_low, lossD_high
    
    

    def _backward_G(self, low_real, high_real, low_fake, high_fake, loss_consistency, loss_IAF, ft_list, img_list):
        self.optimizer.zero_grad()

        low_fake_ft, high_fake_ft = ft_list[2:]
        low_fake_img, high_fake_img = img_list[2:]

        # Train Generator
        label = True
        judge_low_fake_ft = self.netD_low(low_fake_ft).view(-1)
        judge_high_fake_ft = self.netD_high(high_fake_ft).view(-1)
        judge_low_fake_img = self.netD_low(low_fake_img).view(-1)
        judge_high_fake_img = self.netD_high(high_fake_img).view(-1)
        lossG_ft = (self.criterion(judge_low_fake_ft, label) + self.criterion(judge_high_fake_ft, label))
        lossG = (self.criterion(judge_low_fake_img, label) + self.criterion(judge_high_fake_img, label))

        loss_guided = 0
        for i in range(4):
            low_real_small, high_real_small, low_fake_small, high_fake_small = self._choose_channel([low_real, high_real, 
                                                                                                      low_fake, high_fake], i)
            loss_guided += (self.vgg_perceptual_loss(low_real_small, high_fake_small) +
                       self.vgg_perceptual_loss(high_real_small, low_fake_small)) * self.guided_vgg_weight / 4

        loss = lossG + lossG_ft + loss_guided + loss_consistency + loss_IAF

        # backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return lossG, lossG_ft, loss_guided, loss_consistency, loss_IAF

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Overriding method for saving checkpoints from base_trainer.py
        Adding the function of saving discriminators.

        :param epoch: current epoch number
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config,
            'state_dict_D_low': self.netD_low.state_dict(),
            'optimizer_D_low': self.optimizer_D_low.state_dict(),
            'state_dict_D_high': self.netD_high.state_dict(),
            'optimizer_D_high': self.optimizer_D_high.state_dict(),
            # 'state_dict_D_low_ft': self.netD_low_ft.state_dict(),
            # 'optimizer_D_low_ft': self.optimizer_D_low_ft.state_dict(),
            # 'state_dict_D_high_ft': self.netD_high_ft.state_dict(),
            # 'optimizer_D_high_ft': self.optimizer_D_high_ft.state_dict()
        }
        filename = str(self.checkpoint_dir / 'checkpoint-latest.pth'.format(epoch))
        torch.save(state, filename)
        if epoch%10 == 0:
            filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
            torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best and epoch >= 100:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

            if self.uval_best is not None:
                shutil.rmtree(str(self.uval_best))
            self.uval_best = self.uval / ("best(" + str(epoch) + ")")
            self.uval_last.rename(self.uval_best)
            self.uval_last = None

    def _resume_checkpoint(self, resume_path):
        """
        Overriding method for resuming from checkpoints in base_trainer.py

        :param resume_path: Checkpoint path to be resumed
        :return checkpoint: Loaded checkpoint, used mainly for reducing duplicate code while overriding
        """
        checkpoint = super()._resume_checkpoint(resume_path)

        # load model & optimizer of all two discriminators
        if checkpoint['config']['loss'] != "GANLoss":
            self.logger.warning("Warning: GAN loss type given in config file is different from that of checkpoint. "
                                "Optimizer and parameters of discriminators are not being resumed.")
        else:
            self.netD_low.load_state_dict(checkpoint['state_dict_D_low'])
            self.netD_high.load_state_dict(checkpoint['state_dict_D_high'])

            if checkpoint['config'].get('continue'):
                self.optimizer_D_low.load_state_dict(checkpoint['optimizer_D_low'])
                self.optimizer_D_high.load_state_dict(checkpoint['optimizer_D_high'])            
            
            # self.netD_low_ft.load_state_dict(checkpoint['state_dict_D_low_ft'])
            # self.optimizer_D_low_ft.load_state_dict(checkpoint['optimizer_D_low_ft'])
            # self.netD_high_ft.load_state_dict(checkpoint['state_dict_D_high_ft'])
            # self.optimizer_D_high_ft.load_state_dict(checkpoint['optimizer_D_high_ft'])
            self.logger.info("Discriminator parameters loaded successfully.")

        return checkpoint
