from pytz import timezone
from datetime import datetime
from tqdm import tqdm

from network.Nonlinear_3DMM import Nonlinear3DMM
from configure_dataset import *
from renderer.rendering_ops import *
from loss import Loss
import log_utils
from settings import CFG, LOSSES


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
        # self.net = Nonlinear3DMM_UNet().to(CFG.device)

        # Basis
        mu_shape, w_shape = load_Basel_basic('shape')
        mu_exp, w_exp = load_Basel_basic('exp')

        self.mean_shape = torch.tensor(mu_shape + mu_exp, dtype=dtype).to(CFG.device)
        self.std_shape = torch.tensor(np.tile(np.array([1e4, 1e4, 1e4]), CFG.vertex_num), dtype=dtype).to(CFG.device)

        self.mean_m = torch.tensor(np.load(join(CFG.dataset_path, 'mean_m.npy')), dtype=dtype).to(CFG.device)
        self.std_m = torch.tensor(np.load(join(CFG.dataset_path, 'std_m.npy')), dtype=dtype).to(CFG.device)

        self.w_shape = torch.tensor(w_shape, dtype=dtype).to(CFG.device)
        self.w_exp = torch.tensor(w_exp, dtype=dtype).to(CFG.device)

        if True:
            self.random_m_samples = []
            self.random_il_samples = []
            self.random_exp_samples = []
        else:
            pass

        """
        general loss
        {
            "m": 5,
            "texture": 0.5, (기본적으로는 * 1, 만약 recon 안쓰면 * 5)
            "reconstruction": 10, (base acsb absc comb  * 1 * 2 * 2 * 0) / 2
            "landmark": 0.02 (base - gt, comb - gt)
            "perceptual": 2,
        }
        
        base loss (regularization 관련)
        {
            "shape": 10,
            "batchwise_white_shading": 10,
            "const_albedo": 10,
            "const_local_albedo": 10,
            "symmetry": 10,
            "shade_mag": 1,
            "smoothness": 5e5
        }
        
        comb loss
        {
            "gradiff": 10,
            "smoothness": 1
        }
        
        res loss
        {
            "res_ssymmetry": 100,
            "residual"
        }
        """

    def rendering(self, inputs, infer):
        input_images = inputs["input_images"]
        input_exp_labels = inputs["input_exp_labels"]
        input_shape_labels = inputs["input_shape_labels"]
        input_masks = inputs["input_masks"]
        input_m_labels = inputs["input_m_labels"]
        input_texture_labels = inputs["input_texture_labels"]

        lv_m = infer["lv_m"]
        lv_il = infer["lv_il"]
        albedo_base = infer["albedo_base"]
        albedo_comb = infer["albedo_comb"]
        shape_1d_comb = infer["shape_1d_comb"]
        shape_1d_base = infer["shape_1d_base"]
        exp_1d_comb = infer["exp_1d_comb"]
        exp_1d_base = infer["exp_1d_base"]
        # exp = infer["exp"]

        m_full = generate_full(lv_m, self.std_m, self.mean_m)
        m_full_gt = generate_full(input_m_labels, self.std_m, self.mean_m)

        shape_full_comb = generate_full((shape_1d_comb + exp_1d_comb), self.std_shape, self.mean_shape)
        shape_full_base = generate_full((shape_1d_base + exp_1d_base), self.std_shape, self.mean_shape)

        # shape_full_base = generate_full(shape_1d_base, self.std_shape, self.mean_shape)
        # shape_full_comb = generate_full(shape_1d_comb, self.std_shape, self.mean_shape)

        shade_base = generate_shade(lv_il, m_full, shape_full_base)
        shade_comb = generate_shade(lv_il, m_full, shape_full_comb)

        tex_base = generate_texture(albedo_base, shade_base)
        tex_mix_ac_sb = generate_texture(albedo_comb, shade_base)
        tex_mix_ab_sc = generate_texture(albedo_base, shade_comb)  # ab = albedo_base, sc = shape_comb
        tex_comb = generate_texture(albedo_comb, shade_comb)

        u_base, v_base, mask_base = warping_flow(m_full, shape_full_base)
        u_comb, v_comb, mask_comb = warping_flow(m_full, shape_full_comb)

        g_img_mask_base = mask_base.unsqueeze(1) * input_masks
        g_img_mask_comb = mask_comb.unsqueeze(1) * input_masks

        g_img_raw_base = rendering_wflow(tex_base, u_base, v_base)
        g_img_raw_ac_sb = rendering_wflow(tex_mix_ac_sb, u_base, v_base)
        g_img_raw_ab_sc = rendering_wflow(tex_mix_ab_sc, u_comb, v_comb)
        g_img_raw_comb = rendering_wflow(tex_comb, u_comb, v_comb)

        g_img_base = apply_mask(g_img_raw_base, g_img_mask_base, input_images)
        g_img_ac_sb = apply_mask(g_img_raw_ac_sb, g_img_mask_base, input_images)
        g_img_ab_sc = apply_mask(g_img_raw_ab_sc, g_img_mask_comb, input_images)
        g_img_comb = apply_mask(g_img_raw_comb, g_img_mask_comb, input_images)

        # ======= gt =======
        shape_full_gt = generate_full((input_shape_labels + input_exp_labels), self.std_shape, self.mean_shape)
        shade_gt = generate_shade(lv_il, m_full_gt, shape_full_gt)
        u_gt, v_gt, mask_gt = warping_flow(m_full_gt, shape_full_gt)
        g_img_gt = rendering_wflow(input_texture_labels, u_gt, v_gt)

        # for debugging
        g_img_shade_base = rendering_wflow(shade_base, u_base, v_base)
        g_img_shade_comb = rendering_wflow(shade_comb, u_comb, v_comb)
        g_img_shade_gt = rendering_wflow(shade_gt, u_gt, v_gt)

        return {
            "shade_base": shade_base.float(),
            "shade_comb": shade_comb.float(),

            "tex_base": tex_base.float(),
            "tex_mix_ac_sb": tex_mix_ac_sb.float(),
            "tex_mix_ab_sc": tex_mix_ab_sc.float(),
            "tex_comb": tex_comb.float(),

            "g_img_base": g_img_base.float(),
            "g_img_mask_base": g_img_mask_base.float(),

            "g_img_ac_sb": g_img_ac_sb.float(),
            "g_img_ab_sc": g_img_ab_sc.float(),

            "g_img_comb": g_img_comb.float(),
            "g_img_mask_comb": g_img_mask_comb.float(),

            "shape_full_gt": shape_full_gt.float(),
            "shade_gt": shade_gt.float().cpu(),
            "mask_gt": mask_gt.float().cpu(),
            "g_img_gt": g_img_gt.float().cpu(),

            # for debugging
            "g_img_raw_base": g_img_raw_base.float().cpu(),
            "g_img_raw_ac_sb": g_img_raw_ac_sb.float().cpu(),
            "g_img_raw_ab_sc": g_img_raw_ab_sc.float().cpu(),
            "g_img_raw_comb": g_img_raw_comb.float().cpu(),

            "g_img_shade_base": g_img_shade_base.float().cpu(),
            "g_img_shade_comb": g_img_shade_comb.float().cpu(),
            "g_img_shade_gt": g_img_shade_gt.float().cpu(),
        }

    def run_model(self, **inputs):
        # lv_m, lv_il, lv_shape, lv_tex, albedo, shape2d, shape1d, exp = self.net(input_images)

        loss_param = {}

        self.loss.time_start("infer")
        infer = self.net(inputs["input_images"])
        self.loss.time_end("infer")

        self.loss.time_start("render")
        renderer_dict = self.rendering(inputs, infer)
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
                self.logger_train.write_loss_images(loss_param)

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
    #     # 'identity': 10,
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

    pretrained_helper = Nonlinear3DMMHelper(LOSSES, decay_per_epoch)
    pretrained_helper.train()


if __name__ == "__main__":
    pretrained_lr_test()
