from functools import reduce
from pytz import timezone
from datetime import datetime
from tqdm import tqdm

from network.Nonlinear_3DMM_redner import Nonlinear3DMM_redner
from configure_dataset_redner import *
from renderer.rendering_ops_redner import *
from loss_redner import Loss
import log_utils
from settings import CFG, LOSSES, init_3dmm_settings


class Nonlinear3DMMHelper:

    def __init__(self, loss_coefficients, decay_per_epoch=None):
        # initialize parameters
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
        self.net = Nonlinear3DMM_redner().to(CFG.device)

        self.random_angle_samples = []
        self.random_trans_samples = []
        self.random_light_samples = []


    def rendering_for_train( self, input_trans, input_angle, input_light, input_exp, input_shape, input_vcolor, input_image, input_mask,
                             lv_trans, lv_angle, lv_il, exp_1d, albedo_base, albedo_comb,
                             shape_1d_base, shape_1d_comb, **kwargs ):
        
        batch_size = lv_trans.shape[0]

        vt2pixel_u = CFG.vt2pixel_u.view((1, 1, -1)).repeat(batch_size, 1, 1)
        vt2pixel_v = CFG.vt2pixel_v.view((1, 1, -1)).repeat(batch_size, 1, 1)

        vcolor_base = make_1d(albedo_base, vt2pixel_u, vt2pixel_v)
        vcolor_base = vcolor_base.view([batch_size, -1, 3])

        vcolor_comb = make_1d(albedo_comb, vt2pixel_u, vt2pixel_v)
        vcolor_comb = vcolor_comb.view([batch_size, -1, 3])
        
        lv_trans_all    = torch.cat([input_trans, lv_trans, lv_trans, lv_trans, lv_trans], dim=0)
        lv_angle_all    = torch.cat([input_angle, lv_angle, lv_angle, lv_angle, lv_angle], dim=0)
        lv_il_all       = torch.cat([input_light, lv_il, lv_il, lv_il, lv_il], dim=0)
        albedo_all      = torch.cat([input_vcolor, vcolor_base, vcolor_base, vcolor_comb, vcolor_comb], dim=0)
        exp_all         = torch.cat([input_exp, exp_1d, exp_1d, exp_1d, exp_1d], dim=0)
        exp_all         = torch.zeros_like(exp_all, device=CFG.device)
        shape_1d_all    = torch.cat([input_shape, shape_1d_base, shape_1d_comb, shape_1d_base, shape_1d_comb], dim=0)
        input_mask_all  = torch.cat([input_mask, input_mask, input_mask, input_mask, input_mask], dim=0)
        input_image_all = torch.cat([input_image, input_image, input_image, input_image, input_image], dim=0)

        results = render_all(lv_trans_all, lv_angle_all, lv_il_all, albedo_all, exp_all, shape_1d_all,
                         input_mask=input_mask_all, input_background=input_image_all)
        results = list(results.items())

        gt = { }
        base = { }
        comb = { }
        mix_ab_sc = { }
        mix_ac_sb = { }
        
        for idx in range(4):    # 4 : base, ab_sc, ac_sb, comb
            key = results[idx][0]
            result = results[idx][1]

            gt[key + '_gt']            = result[0 * batch_size:1 * batch_size]
            base[key + '_base']        = result[1 * batch_size:2 * batch_size]
            mix_ab_sc[key + '_ab_sc']  = result[2 * batch_size:3 * batch_size]
            mix_ac_sb[key + '_ac_sb']  = result[3 * batch_size:4 * batch_size]
            comb[key + '_comb']        = result[4 * batch_size:5 * batch_size]
            
        return {**gt, **base, **comb, **mix_ac_sb, **mix_ab_sc}


    def run_model(self, **inputs):

        loss_param = {}

        self.loss.time_start("infer")
        infer = self.net(inputs["input_image"])
        self.loss.time_end("infer")

        self.loss.time_start("render")
        renderer_dict = self.rendering_for_train(**{**infer, **inputs})
        self.loss.time_end("render")

        # mask_dict = generate_tex_mask(inputs["input_texture_label"], inputs["input_texture_mask"])

        loss_param.update(inputs)
        loss_param.update(infer)
        loss_param.update(renderer_dict)
        # loss_param.update(mask_dict)

        return loss_param

    def train(self, batch_size=CFG.batch_size):
        # Load datasets
        train_dataset = NonlinearDataset(phase='train', frac=CFG.train_dataset_frac)
        # valid_dataset = NonlinearDataset(phase='valid', frac=CFG.valid_dataset_frac)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True,
                                      num_workers=1, pin_memory=True)

        # valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, drop_last=True, shuffle=False,
        #                               num_workers=1, pin_memory=True)

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
            self.loss.time_start("data_fetching")
            for idx, samples in enumerate(train_dataloader, 0):
                self.loss.time_end("data_fetching")
                self.loss.time_start("total")
                loss_param = self.run_model(**self.sample_to_param(samples))

                g_loss, g_loss_with_landmark = self.loss(**loss_param)

                self.loss.time_start("optimizer")
                if idx % 2 == 0:
                    global_optimizer.zero_grad()
                    g_loss.backward(retain_graph=True)
                    global_optimizer.step()
                else:
                    encoder_optimizer.zero_grad()
                    g_loss_with_landmark.backward(retain_graph=True)
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
            # "input_image_name": samples["image_name"],
            "input_image": samples["image"].to(CFG.device),
            "input_mask": samples["mask"].to(CFG.device),
            
            "input_trans": samples["trans"].to(CFG.device),
            "input_angle": samples["angle"].to(CFG.device),
            "input_light": samples["light"].to(CFG.device),
            "input_exp": samples["exp"].to(CFG.device),
        
            "input_shape": samples["shape"].to(CFG.device),
            "input_vcolor" : samples["vcolor"].to(CFG.device),

            "input_albedo_indexes": list(map(lambda a: a.to(CFG.device), samples["albedo_indices"])),
        }


def pretrained_lr_test(name=None, start_epoch=-1):
    init_3dmm_settings()
    pretrained_helper = Nonlinear3DMMHelper(LOSSES)
    pretrained_helper.train()


if __name__ == "__main__":
    pretrained_lr_test()
