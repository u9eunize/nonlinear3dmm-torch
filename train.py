from network.Nonlinear_3DMM import Nonlinear3DMM
from configure_dataset import *
from pytz import timezone
from datetime import datetime
from renderer.rendering_ops import *
from torch.utils.tensorboard import SummaryWriter
from loss import Loss
import config
from os.path import join




class Nonlinear3DMMHelper:

    def __init__(self, losses):
        # initialize parameters
        dtype = torch.float32
        self.losses = losses
        self.name = f'{datetime.now(timezone("Asia/Seoul")).strftime("%Y%m%d_%H%M%S")}'
        if config.PREFIX:
            self.name += f'_{config.PREFIX}'
        self.writer = SummaryWriter(join(config.LOG_PATH, self.name))
        self.state_file_root_name = join(config.CHECKPOINT_DIR_PATH, self.name)

        # Define losses
        self.loss = Loss(self.losses, self.writer)

        # Load model
        self.net = Nonlinear3DMM().to(config.DEVICE)


        # Basis
        mu_shape, w_shape = load_Basel_basic('shape')
        mu_exp, w_exp = load_Basel_basic('exp')

        self.mean_shape = torch.tensor(mu_shape + mu_exp, dtype=dtype).to(config.DEVICE)
        self.std_shape = torch.tensor(np.tile(np.array([1e4, 1e4, 1e4]), config.VERTEX_NUM), dtype=dtype).to(config.DEVICE)

        self.mean_m = torch.tensor(np.load(join(config.DATASET_PATH, 'mean_m.npy')), dtype=dtype).to(config.DEVICE)
        self.std_m = torch.tensor(np.load(join(config.DATASET_PATH, 'mean_m.npy')), dtype=dtype).to(config.DEVICE)

        self.w_shape = torch.tensor(w_shape, dtype=dtype).to(config.DEVICE)
        self.w_exp = torch.tensor(w_exp, dtype=dtype).to(config.DEVICE)







    def train(self):
        # Load datasets
        train_dataloader = DataLoader(NonlinearDataset(phase='train', frac=config.DATASET_FRAC),
                                      batch_size=config.BATCH_SIZE,
                                      drop_last=True,
                                      shuffle=True,
                                      num_workers=1,
                                      pin_memory=True)

        valid_dataloader = DataLoader(NonlinearDataset(phase='valid', frac=config.DATASET_FRAC),
                                      batch_size=config.BATCH_SIZE,
                                      drop_last=True,
                                      shuffle=False,
                                      num_workers=1,
                                      pin_memory=True)

        # test_dataloader = DataLoader(NonlinearDataset(phase='test', frac=config.DATASET_FRAC),
        #                              batch_size=config.BATCH_SIZE,
        #                              drop_last=True,
        #                              shuffle=False,
        #                              num_workers=1,
        #                              pin_memory=True)


        # Set optimizers
        encoder_optimizer = torch.optim.Adam(self.net.nl_encoder.parameters(), lr=config.LEARNING_RATE, betas=config.BETAS)
        global_optimizer = torch.optim.Adam(self.net.parameters(), lr=config.LEARNING_RATE, betas=config.BETAS)

        # Load checkpoint
        self.net, global_optimizer, encoder_optimizer, start_epoch = load(
            self.net, global_optimizer, encoder_optimizer, start_epoch=config.CHECKPOINT_EPOCH
        )
        global_step = start_epoch * len(train_dataloader) + 1

        # Write graph to the tensorboard
        _, samples = next(enumerate(train_dataloader, 0))
        self.writer.add_graph(self.net, samples["image"].to(config.DEVICE))




        for epoch in range(start_epoch, config.EPOCH):
            # For each batch in the dataloader
            for idx, samples in enumerate(train_dataloader, 0):
                self.train_step(
                    input_images=samples["image"].to(config.DEVICE),
                    input_masks=samples["mask_img"].to(config.DEVICE),
                    input_texture_labels=samples["texture"].to(config.DEVICE),
                    input_texture_masks=samples["mask"].to(config.DEVICE),
                    input_m_labels=samples["m_label"].to(config.DEVICE),
                    input_shape_labels=samples["shape_label"].to(config.DEVICE),
                    input_albedo_indexes=list(map(lambda a: a.to(config.DEVICE), samples["albedo_indices"]))
                )

                if global_step % config.LOSS_LOG_INTERVAL == 0:
                    self.loss.write_losses(global_step, 'train')

                if global_step % config.IMAGE_LOG_INTERVAL == 0:
                    self.loss.write_images(global_step, 'train')


                if idx % 2 == 0:
                    global_optimizer.zero_grad()
                    self.loss.losses['g_loss'].backward()
                    global_optimizer.step()
                else:
                    encoder_optimizer.zero_grad()
                    self.loss.losses['g_loss_with_landmark'].backward()
                    encoder_optimizer.step()


                print(datetime.now(), end=" ")
                print(f"[{epoch}, {idx+1:04d}] {idx * config.BATCH_SIZE}/{len(train_dataloader) * config.BATCH_SIZE} "
                      f"({(idx * config.BATCH_SIZE)/(len(train_dataloader)) * 100:.2f}%) "
                      f"g_loss: {self.loss.losses['g_loss']:.6f}, "
                      f"landmark_loss: {self.loss.losses['g_loss_with_landmark']:.6f}")


                global_step += 1


            save(self.net, global_optimizer, encoder_optimizer, epoch, self.state_file_root_name)





    def train_step(self, **inputs):
        """
        input_albedo_indexes = [x1,y1,x2,y2]
        """
        input_images = inputs["input_images"]

        loss_param = {}

        lv_m, lv_il, lv_shape, lv_tex, albedo, shape2d, shape1d = self.net(input_images)
        renderer_dict = renderer(lv_m, lv_il, albedo, shape2d, shape1d, inputs, self.std_m, self.mean_m, self.std_shape, self.mean_shape)

        network_result = {
            "lv_m": lv_m,
            "lv_il": lv_il,
            "lv_shape": lv_shape,
            "lv_tex": lv_tex,
            "albedo": albedo,
            "shape2d": shape2d,
            "shape1d": shape1d
        }

        loss_param.update(inputs)
        loss_param.update(network_result)
        loss_param.update(renderer_dict)

        return self.loss(**loss_param)








def pretrained_lr_test(name=None, start_epoch=-1):
    losses = [
        'm',
        'shape',
        'landmark',
        'batchwise_white_shading',
        'texture',
        'symmetry',
        'const_albedo',
        'smoothness'
    ]

    pretrained_helper = Nonlinear3DMMHelper(losses)
    pretrained_helper.train()


if __name__ == "__main__":
    pretrained_lr_test()
