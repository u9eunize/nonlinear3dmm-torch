from functools import reduce
from pytz import timezone
from datetime import datetime
from tqdm import tqdm

from network.Nonlinear_3DMM import Nonlinear3DMM
from configure_dataset import *
from renderer.rendering_ops import *
from loss import Loss
import log_utils
from settings import CFG, LOSSES, init_3dmm_settings


class Nonlinear3DMMHelper:

    def __init__(self, loss_coefficients, decay_per_epoch=None):
        # initialize parameters
        dtype = torch.float32
        self.losses = loss_coefficients

        if decay_per_epoch is None:
            decay_per_epoch = dict()
        self.decay_per_epoch = decay_per_epoch

        self.name = f'{datetime.now(timezone("Asia/Seoul")).strftime("%Y%m%d/%H%M%S")}'
        if CFG.name:
            self.name += f'_{CFG.name}'

        # Set Logger
        # self.writer = SummaryWriter(join(CFG.log_path, self.name))

        self.logger_train = log_utils.NLLogger(self.name, "train")
        log_utils.set_logger("nl_train", self.logger_train)

        if CFG.valid:
            self.logger_valid = log_utils.NLLogger(self.name, "valid")
            log_utils.set_logger("nl_valid", self.logger_valid)

        self.state_file_root_name = join(CFG.checkpoint_root_path, self.name)

        # Define losses
        self.loss = Loss(loss_coefficients, decay_per_epoch)

        # Load model
        self.net = Nonlinear3DMM().to(CFG.device)

        if True:
            self.random_m_samples = []
            self.random_il_samples = []
            self.random_exp_samples = []
        else:
            pass

    def rendering_for_train(self, lv_m, lv_il, albedo_base, albedo_comb,
                            shape_1d_base, shape_1d_comb, exp_1d_base, exp_1d_comb,
                            input_images, input_masks, **kwargs):

        base = render_all(lv_m, lv_il, albedo_base, shape_1d_base, exp_1d_base,
                          input_mask=input_masks, input_background=input_images, post_fix="_base")
        comb = render_all(lv_m, lv_il, albedo_comb, shape_1d_comb, exp_1d_comb,
                          input_mask=input_masks, input_background=input_images, post_fix="_comb")

        mix = render_mix(albedo_base, base["shade_base"], albedo_comb, comb["shade_comb"],
                         base["u_base"], base["v_base"], comb["u_comb"], comb["v_comb"],
                         mask_base=base["g_img_mask_base"], mask_comb=comb["g_img_mask_comb"],
                         input_mask=input_masks, input_background=input_images)

        return {**base, **comb, **mix}

    def run_model(self, **inputs):
        # lv_m, lv_il, lv_shape, lv_tex, albedo, shape2d, shape1d, exp = self.net(input_images)

        loss_param = {}

        self.loss.time_start("infer")
        infer = self.net(inputs["input_images"])
        self.loss.time_end("infer")

        self.loss.time_start("render")
        renderer_dict = self.rendering_for_train(**{**infer, **inputs})
        self.loss.time_end("render")

        mask_dict = generate_tex_mask(inputs["input_texture_labels"], inputs["input_texture_masks"])

        loss_param.update(inputs)
        loss_param.update(infer)
        loss_param.update(renderer_dict)
        loss_param.update(mask_dict)

        return loss_param

    def train(self, batch_size=CFG.batch_size):
        # Load datasets
        train_dataset = NonlinearDataset(phase='train', frac=CFG.train_dataset_frac)
        valid_dataset = NonlinearDataset(phase='valid', frac=CFG.valid_dataset_frac)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True,
                                      num_workers=1, pin_memory=True)

        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, drop_last=True, shuffle=False,
                                      num_workers=1, pin_memory=True)

        # Set optimizers
        encoder_optimizer = torch.optim.Adam(self.net.nl_encoder.parameters(),
                                             lr=CFG.lr, betas=CFG.betas)
        global_optimizer = torch.optim.Adam(self.net.parameters(),
                                            lr=CFG.lr, betas=CFG.betas)

        # Load checkpoint
        self.net, global_optimizer, encoder_optimizer, start_epoch, start_step = load(
            self.net, global_optimizer, encoder_optimizer
        )

        if start_step == 0:
            start_step = start_epoch * len(train_dataloader)

        start_step += 1

        self.logger_train.step(start_step)

        # Write graph to the tensorboard
        # _, samples = next(enumerate(train_dataloader, 0))
        # self.writer.add_graph(self.net, samples["image"].to(CFG.device))

        save_per = int(CFG.save_ratio * len(train_dataloader))
        iter_size = len(train_dataloader)

        for epoch in range(start_epoch, CFG.epoch):
            # For each batch in the dataloader
            # camera = []
            # il = []
            # exp = []

            self.loss.time_start("data_fetching")
            for idx, samples in enumerate(train_dataloader, 0):
                self.loss.time_end("data_fetching")
                self.loss.time_start("total")
                loss_param = self.run_model(**self.sample_to_param(samples))

                # camera += loss_param['lv_m'].detach().cpu()
                # il += loss_param['lv_il'].detach().cpu()
                # exp += loss_param['exp'].detach().cpu()

                g_loss, g_loss_with_landmark = self.loss(**loss_param)

                self.loss.time_start("optimizer")
                if idx % 2 == 0:
                    global_optimizer.zero_grad()
                    g_loss.backward()
                    global_optimizer.step()
                else:
                    encoder_optimizer.zero_grad()
                    g_loss_with_landmark.backward()
                    encoder_optimizer.step()
                self.loss.time_end("optimizer")

                self.loss.time_end("total")
                log_utils.NLLogger.print_iteration_log(epoch, self.logger_train.get_step(), idx, batch_size, iter_size)
                log_utils.NLLogger.print_loss_log(self.loss)

                self.loss.time_start("set_log")
                self.logger_train.write_loss_scalar(self.loss)
                self.logger_train.write_loss_images_lazy(loss_param)

                if self.logger_train.get_step() % save_per == 0:
                    save_epoch = epoch
                    if idx == len(train_dataloader) - 1:
                        save_epoch += 1
                    save(self.net, global_optimizer, encoder_optimizer, save_epoch,
                         self.state_file_root_name, self.logger_train.get_step())

                    self.logger_train.save_to_files(self.state_file_root_name, save_epoch)
                    self.logger_train.step()

                    if CFG.valid:
                        self.validate(valid_dataloader, epoch, self.logger_train.get_step())

                    # np.save(f'samples/camera_{epoch}_{idx}', torch.stack(camera, dim=0).numpy())
                    # np.save(f'samples/il_{epoch}_{idx}', torch.stack(il, dim=0).numpy())
                    # np.save(f'samples/exp_{epoch}_{idx}', torch.stack(exp, dim=0).numpy())
                    # camera = []
                    # il = []
                    # exp = []

                else:
                    self.logger_train.step()

                self.loss.time_end("set_log")
                self.loss.time_start("empty_cache")
                torch.cuda.empty_cache()
                self.loss.time_end("empty_cache")
                self.loss.time_start("data_fetching")
            if epoch % 2 == 0:
                self.loss.decay_coefficient()

    def validate(self, valid_dataloader, epoch, global_step):
        print("\n\n", "*" * 10, "start validation", "*" * 10, "\n")

        self.logger_valid.step(global_step, is_flush=False)

        loss_param, loss_avg, loss_max, loss_min = self.test(valid_dataloader, epoch=epoch, step=global_step)

        for loss_name, loss in loss_avg.items():
            self.logger_valid.write_scalar(loss_name, loss, interval=1)
        self.logger_valid.write_loss_images(loss_param, interval=1)

        self.logger_valid.step(global_step)

        print("\n\n", "*" * 10, "end validation", "*" * 10, "\n")

    def test(self, dataloader, load_model=False, load_dataset=False, batch_size=CFG.batch_size,
             epoch=CFG.checkpoint_epoch, step=CFG.checkpoint_step):
        if load_dataset:
            dataset = NonlinearDataset(phase='test', frac=CFG.test_dataset_frac)
            dataloader = DataLoader(dataset, batch_size=batch_size,
                                    drop_last=True, shuffle=False, num_workers=1, pin_memory=True)
        if load_model:
            self.net, _, _, start_epoch, start_step = load(self.net, start_epoch=epoch, start_step=step)

        with torch.no_grad():
            loss_param = dict()
            loss_avg = dict()
            loss_max = dict()
            loss_min = dict()

            print("total dataset size :", len(dataloader) * batch_size)

            for idx, samples in enumerate(tqdm(dataloader), 0):
                loss_param = self.run_model(**self.sample_to_param(samples))
                self.loss(**loss_param)

                for key, loss_value in self.loss.losses.items():
                    # loss_value = loss_value.item()

                    if key not in loss_avg:
                        loss_avg[key] = loss_value
                        loss_max[key] = loss_value
                        loss_min[key] = loss_value
                    else:
                        loss_avg[key] += loss_value
                        loss_max[key] = max(loss_max[key], loss_value)
                        loss_min[key] = min(loss_min[key], loss_value)

            print()
            for key in self.loss.losses.keys():
                loss_avg[key] /= len(dataloader)
                print(key, f"(avg: {loss_avg[key]:02f}, max: {loss_max[key]:02f}, min: {loss_min[key]:02f})")

        return loss_param, loss_avg, loss_max, loss_min

    def sample_to_param(self, samples):
        return {
            "input_images": samples["image"].to(CFG.device),
            "input_masks": samples["mask_img"].to(CFG.device),
            "input_texture_labels": samples["texture"].to(CFG.device),
            "input_texture_masks": samples["mask"].to(CFG.device),
            "input_m_labels": samples["m_label"].to(CFG.device),
            "input_shape_labels": samples["shape_label"].to(CFG.device),
            "input_albedo_indexes": list(map(lambda a: a.to(CFG.device), samples["albedo_indices"])),
            "input_exp_labels": samples["exp_label"].to(CFG.device)
        }


def pretrained_lr_test(name=None, start_epoch=-1):
    # losses = {
    #     'm': 5,  # origin: 5
    #     'shape': 10,  # origin: 10
    #     # 'expression': 10,  # origin: 10
    #     'batchwise_white_shading': 10,  # origin: 10
    #
    #     'base_landmark': 0.02,  # origin: 0.02
    #     'comb_landmark': 0.02,  # origin: 0.02
    #     'gt_landmark': 0.02,  # origin: 0.02
    #
    #     'base_texture': 2,  # if using texture loss using 0.5, else 2.5
    #     'mix_ac_sb_texture': 2,  # if using texture loss  using 0.5, else 2.5
    #     'mix_ab_sc_texture': 2,  # if using texture loss  using 0.5, else 2.5
    #     # 'comb_texture': 0,  # if using texture loss  using 0.5, else 2.5
    #
    #     # 'base_perceptual_recon': 0,    # default 0
    #     'mix_ab_sc_perceptual_recon': 10 * 100,  # default 10
    #     'mix_ac_sb_perceptual_recon': 10 * 100,  # default 10
    #     # 'comb_perceptual_recon': 0,
    #
    #     'base_pix_recon': 10 / 2,
    #     'mix_ab_sc_pix_recon': 10 * 2 / 2,
    #     'mix_ac_sb_pix_recon': 10 * 2 / 2,
    #     # 'comb_pix_recon': 0,
    #
    #     'symmetry': 10,  # origin: 10
    #     'const_albedo': 10,  # origin: 10
    #     # 'shade_mag': 1,  # origin: 1
    #
    #     'base_smoothness': 5e5,  # origin: 5e5
    #     'comb_smoothness': 1,  # origin: 1
    #
    #     #'shape_residual': 1,  # origin: 1
    #     #'albedo_residual': 1,  # origin: 1
    #
    #     # 'identity': 10,f
    #     # 'content': 10,
    #     # 'gradient_difference': 10,
    # }
    decay_per_epoch = {
        # 'm': 0.8,
        # 'shape': 0.8,
        # 'base_texture': 0.8,
        # 'mix_ac_sb_texture': 0.8,
        # 'mix_ab_sc_texture': 0.8,
    }

    init_3dmm_settings()

    pretrained_helper = Nonlinear3DMMHelper(LOSSES, decay_per_epoch)
    pretrained_helper.train()


if __name__ == "__main__":
    pretrained_lr_test()
