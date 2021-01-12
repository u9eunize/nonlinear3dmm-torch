from os.path import join
from pytz import timezone
from datetime import datetime

import torch
import torchvision
from torchvision.utils import *
from torch.utils.tensorboard import SummaryWriter
from renderer.rendering_ops import generate_full
from utils import get_checkpoint_dir, save_configuration
from renderer.rendering_ops import *

from settings import CFG


class NLLogger:
    def __init__(self, name, phase, start_step=1, log_image_count=CFG.log_image_count):
        self.path = join(CFG.log_path, f"{name}_{phase}")
        self.writer = SummaryWriter(self.path)

        save_configuration(self.path)
        self._step = start_step
        self.holder = {}
        self.log_image_count = log_image_count
        self.lazy_params = dict()

    def step(self, step=None, is_flush=True):
        if step is not None:
            self._step = step
        else:
            self._step += 1

        if is_flush:
            # do flush
            use = []
            next_lazy_params = dict()
            for interval, lazy_args in self.lazy_params.items():
                if self._step % interval == 0:
                    self.write_loss_images(**lazy_args)
                else:
                    next_lazy_params[interval] = lazy_args
            self.lazy_params = next_lazy_params
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

    def write_scalar(self, name, data, interval=CFG.log_loss_interval):
        self._write(interval, f"{name}", (NLLogger.add_scalar, data))

    def write_image(self, name, images, interval=CFG.log_image_interval):
        if images is None:
            return
        if not isinstance(images, list):
            images = [images]
        result = []
        for i in range(self.log_image_count):
            r = torch.cat([img[i:i+1, :, :, :] for img in images], dim=0)
            result.append(r)

        row_limit = (8 // len(images)) * len(images)
        result = torchvision.utils.make_grid(torch.cat(result, dim=0), nrow=row_limit).unsqueeze(0)
        # save_image(result, f'./logs/{self.get_step()}_{name}.jpg')
        self._write(interval, f"{name}", (NLLogger.add_images, result))

    def write_mesh(self, name, data, interval=CFG.log_image_interval):
        vertices = data["vertices"]
        vertices = vertices + CFG.mean_shape
        vertices = vertices.view((CFG.batch_size, -1, 3))
        data["vertices"] = vertices[:1, :, :].clone().cpu()
        # data["faces"] = CFG.face[:1, :CFG.tri_num, :].clone().cpu()
        data["faces"] = CFG.face.unsqueeze(0).clone().cpu() - 1
        self._write(interval, f"{name}", (NLLogger.add_mesh, data))

    def save_to_files(self, path, epoch):
        for interval, logs in self.holder.items():
            for name, flusher in logs.items():
                fn, data = flusher
                filename = join(get_checkpoint_dir(path, epoch), f"{name}_{self._step}.png")
                if fn == NLLogger.add_images:
                    torchvision.utils.save_image(data, filename)

    def write_func(self, name, func, data, interval=CFG.log_image_interval):
        self._write(interval, name, (func, data))

    def write_loss_scalar(self, loss, interval=CFG.log_image_interval):
        for key, loss_value in loss.losses.items():
            self.write_scalar(key, loss_value)

    @staticmethod
    def match_size(origin, vec):
        ret = torch.zeros(origin.shape).cuda()
        ret[:, :, :vec.shape[2], :] = vec
        return ret

    def _write_loss_images(self, name, loss_params, keywords, img_sz=None, interval=CFG.log_image_interval):
        if img_sz is None:
            img_sz = loss_params[keywords[0]]
        img_list = []
        for key in keywords:
            img = self.match_size(img_sz, loss_params[key])
            img = img.clamp(0, 1) if "shade" not in key else img.clamp(-1, 1)
            img_list.append(img.cpu())
        self.write_image(name, img_list, interval=interval)


    @staticmethod
    def rendering_for_log(input_images, input_m_labels, input_shape_labels, input_exp_labels,
                          input_texture_labels, input_masks,
                          lv_m, lv_il, albedo_base, albedo_comb,
                          shape_1d_base, shape_1d_comb, exp_1d_base, exp_1d_comb, **kwargs):
        m_full_gt = generate_full(input_m_labels, "m")
        shape_full_gt = generate_full(input_shape_labels, "shape")
        exp_full_gt = generate_full(input_exp_labels, "exp")

        shade_gt = generate_shade(lv_il, m_full_gt, shape_full_gt)
        u_gt, v_gt, mask_gt = warping_flow(m_full_gt, shape_full_gt)
        g_img_gt = rendering_wflow(input_texture_labels, u_gt, v_gt).cpu()

        shade_exp_gt = generate_shade(lv_il, m_full_gt, shape_full_gt + exp_full_gt)
        u_exp_gt, v_exp_gt, mask_exp_gt = warping_flow(m_full_gt, shape_full_gt + exp_full_gt)
        g_img_exp_gt = rendering_wflow(input_texture_labels, u_exp_gt, v_exp_gt).cpu()

        base_exp = render_all(lv_m, lv_il, albedo_base, shape_1d_base, exp_1d_base, using_expression=True)
        base_exp_raw = base_exp["g_img"].cpu()
        base_exp_mask = apply_mask(base_exp["g_img"], base_exp["g_img_mask"] * input_masks, input_images).cpu()
        comb_exp = render_all(lv_m, lv_il, albedo_comb, shape_1d_comb, exp_1d_comb, using_expression=True)
        comb_exp_raw = comb_exp["g_img"].cpu()
        comb_exp_mask = apply_mask(comb_exp["g_img"], comb_exp["g_img_mask"] * input_masks, input_images).cpu()
        base_no_exp = render_all(lv_m, lv_il, albedo_base, shape_1d_base, exp_1d_base, using_expression=False)
        base_no_exp_raw = base_no_exp["g_img"].cpu()
        base_no_exp_mask = apply_mask(base_no_exp["g_img"], base_no_exp["g_img_mask"] * input_masks, input_images).cpu()
        comb_no_exp = render_all(lv_m, lv_il, albedo_comb, shape_1d_comb, exp_1d_comb, using_expression=False)
        comb_no_exp_raw = comb_no_exp["g_img"].cpu()
        comb_no_exp_mask = apply_mask(comb_no_exp["g_img"], comb_no_exp["g_img_mask"] * input_masks, input_images).cpu()

        return {
            "shade_gt": shade_gt.float().cpu(),
            "mask_gt": mask_gt.float().cpu(),
            "g_img_gt": g_img_gt.float().cpu(),

            "shade_exp_gt": shade_exp_gt.float().cpu(),
            "mask_exp_gt": mask_exp_gt.float().cpu(),
            "g_img_exp_gt": g_img_exp_gt.float().cpu(),

            "base_exp_raw": base_exp_raw.cpu(),
            "base_exp_mask": base_exp_mask.cpu(),
            "comb_exp_raw": comb_exp_raw.cpu(),
            "comb_exp_mask": comb_exp_mask.cpu(),
            "base_no_exp_raw": base_no_exp_raw.cpu(),
            "base_no_exp_mask": base_no_exp_mask.cpu(),
            "comb_no_exp_raw": comb_no_exp_raw.cpu(),
            "comb_no_exp_mask": comb_no_exp_mask.cpu(),
        }

    def write_loss_images_lazy(self, loss_params, interval=CFG.log_image_interval):
        self.lazy_params[interval] = {
            "loss_params": loss_params,
            "interval": interval,
        }

    def write_loss_images(self, loss_params, interval=CFG.log_image_interval):
        all_dict = dict()
        all_dict.update(loss_params)
        # all_dict.update(NLLogger.rendering_for_log(**loss_params))
        # self.write_image("shade", loss_params["shade"], interval=interval)
        self._write_loss_images("g_images", all_dict, [
            # "input_image",
            # "g_img_bg_gt",
            # "g_img_bg_base",
            # "g_img_bg_ac_sb",
            # "g_img_bg_ab_sc",
            # "g_img_bg_comb",

            "input_image",
            "g_img_bg_base",

            # "input_images",
            # "g_img_base",
            # "g_img_ac_sb",
            # "g_img_ab_sc",
            # "g_img_comb",
            # "g_img_gt",
            # "g_img_exp_gt",
        ], interval=interval)
        self._write_loss_images("g_albedo", all_dict, [
            "albedo_base",
            "albedo_comb",
        ])
        # self._write_loss_images("g_image_exp", all_dict, [
        #     "base_exp_raw",
        #     "base_no_exp_raw",
        #     "comb_exp_raw",
        #     "comb_no_exp_raw",
        #     "base_exp_mask",
        #     "base_no_exp_mask",
        #     "comb_exp_mask",
        #     "comb_no_exp_mask",
        # ], interval=interval)

        # self._write_loss_images("albedo_and_shade", all_dict, [
        #     "input_texture_labels",
        #     "albedo_base",
        #     "albedo_comb",
        #     "shade_gt",
        #     "shade_base",
        #     "shade_comb",
        # ], interval=interval)

        # self._write_loss_images("texture", all_dict, [
        #     "input_texture_labels",
        #     "tex_base",
        #     "tex_mix_ac_sb",
        #     "tex_mix_ab_sc",
        # ], interval=interval)

        # self.write_mesh("gt_shape_mesh", {
        #     "vertices": all_dict["input_shape_labels"] + all_dict["input_exp_labels"],
        # }, interval=interval)
        #
        # self.write_mesh("gt_shape_mesh", {
        #     "vertices": all_dict["input_shape"] + all_dict["input_exp"],
        # }, interval=interval)
        self.write_mesh("base_shape_mesh", {
            "vertices": all_dict["shape_1d_base"],
        }, interval=interval)
        # self.write_mesh("base_shape_exp_mesh", {
        #     "vertices": all_dict["shape_1d_base"] + all_dict["exp_1d"],
        # }, interval=interval)
        # self.write_mesh("comb_shape_exp_mesh", {
        #     "vertices": all_dict["shape_1d_comb"] + all_dict["exp_1d"],
        # }, interval=interval)

        # if CFG.using_expression:
        #     self.write_mesh("base_shape_mesh_with_exp", {
        #         "vertices": all_dict["shape_1d_base"] + all_dict["exp_1d_base"],
        #     }, interval=interval)
        #
        #     self.write_mesh("comb_shape_mesh_with_exp", {
        #         "vertices": all_dict["shape_1d_comb"] + all_dict["exp_1d_comb"],
        #     }, interval=interval)

        # self.write_image("g_images_rand", [
        #     loss_params["input_images"],
        #     loss_params["g_images"],
        #     loss_params["g_images_random"],
        #     loss_params["g_images_mask_random"],
        #     self.match_size(loss_params["input_images"], (loss_params["albedo"] + 1) / 2),
        #     self.match_size(loss_params["input_images"], loss_params["random_shade"]),
        #     self.match_size(loss_params["input_images"], loss_params["random_tex"]),
        #     self.match_size(loss_params["input_images"], loss_params["input_texture_labels"]),
        # ], interval=interval)

        # self.write_image("texture", [
        #     loss_params["tex"],
        #     loss_params["input_texture_labels"],
        #     loss_params["tex"] * loss_params["tex_vis_mask"],
        #     loss_params["input_texture_labels"] * loss_params["tex_vis_mask"]
        # ], interval=interval)

        # self.write_image("texture", [
        #     loss_params["tex"],
        #     loss_params["input_texture_labels"],
        #     loss_params["tex"] * loss_params["tex_vis_mask"],
        #     loss_params["input_texture_labels"] * loss_params["tex_vis_mask"]
        # ], interval=interval)

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
            loss_name = key.replace("_loss", "")
            print_name = loss_name
            if loss_name in loss.decay_per_epoch:
                print_name = print_name + f"({loss.decay_per_epoch[loss_name] ** loss.decay_step:.2f})"
            print(print_name, ":",  f"{loss_value.item():.4f}", end=" ")
        print()
        if CFG.verbose == "debug":
            for key, time_value in loss.time_checker.items():
                print(key + ":", f"{time_value:.2f}", end=" ")
        print("total:", f"{loss.time_checker['total']:.2f}", "ms")

    @staticmethod
    def add_images(writer, name, data, step):
        writer.add_images(name, data, step)

    @staticmethod
    def add_scalar(writer, name, data, step):
        writer.add_scalar(name, data, step)

    @staticmethod
    def add_mesh(writer, name, data, step):
        writer.add_mesh(name, **data, global_step=step)




_LOG_DICT = dict()


def set_logger(identifier, logger):
    _LOG_DICT[identifier] = logger


def get_logger(identifier):
    return _LOG_DICT[identifier]
