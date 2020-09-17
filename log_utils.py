from os.path import join
from pytz import timezone
from datetime import datetime

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

import config


class NLLogger:
    def __init__(self, name, phase, start_step=0, img_log_number=config.IMAGE_LOG_NUMBER):
        self.writer = SummaryWriter(join(config.LOG_PATH, f"{name}_{phase}"))
        self._step = start_step
        self.holder = {}
        self.img_log_number = img_log_number

    def step(self, step=None, is_flush=True):
        if step is not None:
            self._step = step
        else:
            self._step += 1

        if is_flush:
            # do flush
            use = []
            for interval, logs in self.holder.items():
                if self._step % interval == 0:
                    use.append(interval)
                    for name, flusher in logs.items():
                        fn, data = flusher
                        fn(self.writer, name, data, self._step)

            # delete used logs
            for interval in use:
                del self.holder[interval]

            self.writer.flush()

    def get_step(self):
        return self._step

    def _write(self, interval, save_name, log_data):
        if interval not in self.holder:
            self.holder[interval] = dict()
        self.holder[interval][save_name] = log_data

    def write_scalar(self, name, data, interval=config.LOSS_LOG_INTERVAL):
        self._write(interval, f"{name}", (NLLogger.add_scalar, data))

    def write_image(self, name, images, interval=config.IMAGE_LOG_INTERVAL):
        if images is None:
            return
        if not isinstance(images, list):
            images = [images]
        result = []
        for i in range(self.img_log_number):
            r = torch.cat([img[i:i+1, :, :, :] for img in images], dim=0)
            result.append(r)

        row_limit = (8 // len(images)) * len(images)
        result = torchvision.utils.make_grid(torch.cat(result, dim=0), nrow=row_limit).unsqueeze(0)

        self._write(interval, f"{name}", (NLLogger.add_images, result))

    def write_func(self, name, func, data, interval=config.IMAGE_LOG_INTERVAL):
        self._write(interval, name, (func, data))

    def write_loss_scalar(self, loss, interval=config.IMAGE_LOG_INTERVAL):
        for key, loss_value in loss.losses.items():
            self.write_scalar(key, loss_value)

    def write_loss_images(self, loss_params, interval=config.IMAGE_LOG_INTERVAL):

        shade = torch.zeros(loss_params["input_images"].shape).cuda()
        shade[:, :, :loss_params["shade"].shape[2], :] = loss_params["shade"]
        self.write_image("shade", loss_params["shade"], interval=interval)
        self.write_image("g_images", [
            loss_params["input_images"],
            loss_params["g_images_gt"],
            loss_params["g_images_raw"],
            loss_params["g_images"],
            loss_params["input_masks"],
            loss_params["g_images_mask_raw"],
            loss_params["g_images_mask"],
            shade,
        ], interval=interval)

        self.write_image("texture", [
            loss_params["tex"],
            loss_params["input_texture_labels"],
            loss_params["tex"] * loss_params["tex_vis_mask"],
            loss_params["input_texture_labels"] * loss_params["tex_vis_mask"]
        ], interval=interval)

    @staticmethod
    def print_iteration_log(epoch, step, idx, batch_size, iteration_size):
        print(datetime.now(timezone("Asia/Seoul")), end=" ")

        print(f"[{epoch}, {idx + 1:04d}, {step}] "
              f"[{idx + 1:04d}] "
              f"{idx * batch_size}/{iteration_size * batch_size} "
              f"({idx / iteration_size * 100:.2f}%) ")

    @staticmethod
    def print_loss_log(loss):
        for key, loss_value in loss.losses.items():
            print(key.replace("_loss", "") + ":", f"{loss_value.item():.4f}", end=" ")
        print()

    @staticmethod
    def add_images(writer, name, data, step):
        writer.add_images(name, data, step)

    @staticmethod
    def add_scalar(writer, name, data, step):
        writer.add_scalar(name, data, step)


_LOG_DICT = dict()


def set_logger(identifier, logger):
    _LOG_DICT[identifier] = logger


def get_logger(identifier):
    return _LOG_DICT[identifier]
