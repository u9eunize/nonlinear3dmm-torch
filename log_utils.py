from os.path import join

import torch
from torch.utils.tensorboard import SummaryWriter

import config


class NLLogger:
    def __init__(self, name, phase, start_step=0, img_log_number=config.IMAGE_LOG_NUMBER):
        self.writer = SummaryWriter(join(config.LOG_PATH, f"{name}_{phase}"))
        self._step = start_step
        self.holder = {}
        self.img_log_number = img_log_number

    def step(self, step=None):
        if step is not None:
            self._step = step
        else:
            self._step += 1

        # do flush
        use = []
        for interval, logs in self.holder.items():
            if self._step % interval == 0:
                use.append(interval)
                for name, flusher in logs.items():
                    fn, data = flusher
                    fn(name, data, self._step)

        # delete used logs
        for interval in use:
            del self.holder[interval]

        self.writer.flush()

    def _write(self, interval, save_name, log_data):
        if interval not in self.holder:
            self.holder[interval] = dict()
        self.holder[interval][save_name] = log_data

    def write_scalar(self, name, data, interval=config.LOSS_LOG_INTERVAL):
        self._write(interval, f"{name}", (self.writer.add_scalar, data))

    def write_image(self, name, images, interval=config.IMAGE_LOG_INTERVAL):
        if images is None:
            return
        if not isinstance(images, list):
            images = [images]
        result = []
        for img in images:
            result.append(img[:self.img_log_number])
        result = torch.cat(result, dim=0)

        self._write(interval, f"{name}", (self.writer.add_images, result))


_LOG_DICT = dict()


def set_logger(identifier, logger):
    _LOG_DICT[identifier] = logger


def get_logger(identifier):
    return _LOG_DICT[identifier]
