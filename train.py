from pytz import timezone
from datetime import datetime
from tqdm import tqdm

from network.Nonlinear_3DMM import Nonlinear3DMM
from configure_dataset import *
from renderer.rendering_ops import *
from loss import Loss
import log_utils
import config


class Nonlinear3DMMHelper:

    def __init__(self, losses):
        # initialize parameters
        dtype = torch.float32
        self.losses = losses
        self.name = f'{datetime.now(timezone("Asia/Seoul")).strftime("%Y%m%d_%H%M%S")}'
        if config.PREFIX:
            self.name += f'_{config.PREFIX}'

        # Set Logger
        # self.writer = SummaryWriter(join(config.LOG_PATH, self.name))

        self.logger_train = log_utils.NLLogger(self.name, "train")
        log_utils.set_logger("nl_train", self.logger_train)

        self.logger_valid = log_utils.NLLogger(self.name, "valid")
        log_utils.set_logger("nl_valid", self.logger_valid)

        self.state_file_root_name = join(config.CHECKPOINT_DIR_PATH, self.name)

        # Define losses
        self.loss = Loss(self.losses)

        # Load model
        self.net = Nonlinear3DMM().to(config.DEVICE)

        # Basis
        mu_shape, w_shape = load_Basel_basic('shape')
        mu_exp, w_exp = load_Basel_basic('exp')

        self.mean_shape = torch.tensor(mu_shape + mu_exp, dtype=dtype).to(config.DEVICE)
        self.std_shape = torch.tensor(np.tile(np.array([1e4, 1e4, 1e4]), config.VERTEX_NUM), dtype=dtype).to(config.DEVICE)

        self.mean_m = torch.tensor(np.load(join(config.DATASET_PATH, 'mean_m.npy')), dtype=dtype).to(config.DEVICE)
        self.std_m = torch.tensor(np.load(join(config.DATASET_PATH, 'std_m.npy')), dtype=dtype).to(config.DEVICE)

        self.w_shape = torch.tensor(w_shape, dtype=dtype).to(config.DEVICE)
        self.w_exp = torch.tensor(w_exp, dtype=dtype).to(config.DEVICE)

        if True:
            self.random_m_samples = []
            self.random_il_samples = []
            self.random_exp_samples = []
        else:
            pass

    def run_model(self, **inputs):
        input_images = inputs["input_images"]
        batch_size = input_images.shape[0]

        loss_param = {}

        lv_m, lv_il, lv_shape, lv_tex, albedo, shape2d, shape1d, exp = self.net(input_images)

        if True:
            self.random_m_samples = [lv_m.detach().cpu()] + self.random_m_samples
            self.random_m_samples = self.random_m_samples[:min(config.RANDOM_SAMPLE_NUM, len(self.random_m_samples))]
            self.random_il_samples = [lv_il.detach().cpu()] + self.random_il_samples
            self.random_il_samples = self.random_il_samples[:min(config.RANDOM_SAMPLE_NUM, len(self.random_il_samples))]
            self.random_exp_samples = [exp.detach().cpu()] + self.random_exp_samples
            self.random_exp_samples = self.random_exp_samples[:min(config.RANDOM_SAMPLE_NUM, len(self.random_exp_samples))]
        else:
            pass


        m_full = generate_full(lv_m, self.std_m, self.mean_m)
        shape_full = generate_full((shape1d + exp), self.std_shape, self.mean_shape)

        gt_m_full = generate_full(inputs["input_m_labels"], self.std_m, self.mean_m)
        # gt_shape_full = generate_full(inputs["input_shape_labels"], self.std_shape, self.mean_shape)
        gt_shape_full = generate_full(inputs["input_shape_labels"] + inputs["input_exp_labels"], self.std_shape, self.mean_shape)

        shade, tex = generate_shade_and_texture(m_full, lv_il, albedo, shape_full)
        # rand_shade, rand_tex = generate_shade_and_texture(m_full, lv_il, albedo, shape_full)

        renderer_dict = renderer(m_full, tex, shape_full, inputs)
        renderer_dict_gt = renderer(gt_m_full, inputs["input_texture_labels"], gt_shape_full, inputs, "_gt")
        # renderer_dict_rand = renderer(rand_m_full, rand_tex, shape_full, inputs, "_rand")

        mask_dict = generate_tex_mask(batch_size, inputs["input_texture_labels"], inputs["input_texture_masks"])


        # Random feeding data
        if config.RANDOM_CAMERA:
            indices = torch.randint(len(self.random_m_samples), (batch_size,))
            random_camera = torch.cat(self.random_m_samples, dim=0)[indices].to(config.DEVICE)
            random_camera_full = generate_full(random_camera, self.std_m, self.mean_m)

        else:
            random_camera_full = m_full
        if config.RANDOM_EXPRESSION:
            indices = torch.randint(len(self.random_exp_samples), (batch_size,))
            random_exp = torch.cat(self.random_exp_samples, dim=0)[indices].to(config.DEVICE)
            random_shape_full = generate_full(shape1d + random_exp, self.std_shape, self.mean_shape)
        else:
            random_shape_full = shape_full
        if config.RANDOM_ILLUMINATION:
            indices = torch.randint(len(self.random_il_samples), (batch_size,))
            random_il = torch.cat(self.random_il_samples, dim=0)[indices].to(config.DEVICE)
        else:
            random_il = lv_il

        random_shade, random_tex = generate_shade_and_texture(random_camera_full, random_il, albedo, random_shape_full)
        renderer_dict_random = renderer_random(random_camera_full, random_tex, random_shape_full, '_random')


        network_result = {
            "lv_m": lv_m,
            "lv_il": lv_il,
            "lv_shape": lv_shape,
            "lv_tex": lv_tex,
            "albedo": albedo,
            "shape2d": shape2d,
            "shape1d": shape1d,
            "exp": exp
        }

        loss_param.update({
            "shade": shade,
            "tex": tex,
        })
        loss_param.update(inputs)
        loss_param.update(network_result)
        loss_param.update(renderer_dict)
        loss_param.update(renderer_dict_gt)
        loss_param.update(mask_dict)

        loss_param.update(renderer_dict_random)
        loss_param.update({
            "random_shade": random_shade,
            "random_tex": random_tex
        })

        return loss_param

    def train(self, batch_size=config.BATCH_SIZE):
        # Load datasets
        train_dataset = NonlinearDataset(phase='train', frac=config.TRAIN_DATASET_FRAC)
        valid_dataset = NonlinearDataset(phase='valid', frac=config.VALID_DATASET_FRAC)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True,
                                      num_workers=1, pin_memory=True)

        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, drop_last=True, shuffle=False,
                                      num_workers=1, pin_memory=True)

        # Set optimizers
        encoder_optimizer = torch.optim.Adam(self.net.nl_encoder.parameters(),
                                             lr=config.LEARNING_RATE, betas=config.BETAS)
        global_optimizer = torch.optim.Adam(self.net.parameters(),
                                            lr=config.LEARNING_RATE, betas=config.BETAS)

        # Load checkpoint
        self.net, global_optimizer, encoder_optimizer, start_epoch, start_step = load(
            self.net, global_optimizer, encoder_optimizer, start_epoch=config.CHECKPOINT_EPOCH
        )

        if start_step == 0:
            start_step = start_epoch * len(train_dataloader)

        start_step += 1

        self.logger_train.step(start_step)

        # Write graph to the tensorboard
        # _, samples = next(enumerate(train_dataloader, 0))
        # self.writer.add_graph(self.net, samples["image"].to(config.DEVICE))

        save_per = int(config.SAVE_PER_RATIO * len(train_dataloader))
        iter_size = len(train_dataloader)


        for epoch in range(start_epoch, config.EPOCH):
            # For each batch in the dataloader
            # camera = []
            # il = []
            # exp = []

            for idx, samples in enumerate(train_dataloader, 0):
                loss_param = self.run_model(**self.sample_to_param(samples))

                # camera += loss_param['lv_m'].detach().cpu()
                # il += loss_param['lv_il'].detach().cpu()
                # exp += loss_param['exp'].detach().cpu()

                g_loss, g_loss_with_landmark = self.loss(**loss_param)

                if idx % 2 == 0:
                    global_optimizer.zero_grad()
                    g_loss.backward()
                    global_optimizer.step()
                else:
                    encoder_optimizer.zero_grad()
                    g_loss_with_landmark.backward()
                    encoder_optimizer.step()

                log_utils.NLLogger.print_iteration_log(epoch, self.logger_train.get_step(), idx, batch_size, iter_size)
                log_utils.NLLogger.print_loss_log(self.loss)

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

                    # self.validate(valid_dataloader, epoch, self.logger_train.get_step())

                    # np.save(f'samples/camera_{epoch}_{idx}', torch.stack(camera, dim=0).numpy())
                    # np.save(f'samples/il_{epoch}_{idx}', torch.stack(il, dim=0).numpy())
                    # np.save(f'samples/exp_{epoch}_{idx}', torch.stack(exp, dim=0).numpy())
                    # camera = []
                    # il = []
                    # exp = []

                else:
                    self.logger_train.step()


    def validate(self, valid_dataloader, epoch, global_step):
        print("\n\n", "*" * 10, "start validation", "*" * 10, "\n")

        self.logger_valid.step(global_step, is_flush=False)

        loss_param, loss_avg, loss_max, loss_min = self.test(valid_dataloader, epoch=epoch, step=global_step)

        for loss_name, loss in loss_avg.items():
            self.logger_valid.write_scalar(loss_name, loss, interval=1)
        self.logger_valid.write_loss_images(loss_param, interval=1)

        self.logger_valid.step(global_step)

        print("\n\n", "*" * 10, "end validation", "*" * 10, "\n")

    def test(self, dataloader, load_model=False, load_dataset=False, batch_size=config.BATCH_SIZE,
             epoch=config.CHECKPOINT_EPOCH, step=config.CHECKPOINT_STEP):
        if load_dataset:
            dataset = NonlinearDataset(phase='test', frac=config.TEST_DATASET_FRAC)
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
                    loss_value = loss_value.item()

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
            "input_images": samples["image"].to(config.DEVICE),
            "input_masks": samples["mask_img"].to(config.DEVICE),
            "input_texture_labels": samples["texture"].to(config.DEVICE),
            "input_texture_masks": samples["mask"].to(config.DEVICE),
            "input_m_labels": samples["m_label"].to(config.DEVICE),
            "input_shape_labels": samples["shape_label"].to(config.DEVICE),
            "input_albedo_indexes": list(map(lambda a: a.to(config.DEVICE), samples["albedo_indices"])),
            "input_exp_labels": samples["exp_label"].to(config.DEVICE)
        }


def pretrained_lr_test(name=None, start_epoch=-1):
    losses = [
        'm',
        'shape',
        'landmark',
        'batchwise_white_shading',
        'texture',
        #'reconstruction',
        'symmetry',
        'const_albedo',
        'smoothness',
        'expression',

        'identity',
        'content',
    ]

    pretrained_helper = Nonlinear3DMMHelper(losses)
    pretrained_helper.train()


if __name__ == "__main__":
    pretrained_lr_test()
