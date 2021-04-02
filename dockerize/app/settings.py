import argparse, random
import sys
import json
import torch
from os.path import join
import numpy as np

LOSS_TYPE_LIST = ["l1", "l2", "l2,1"]


def parse():
    parser = argparse.ArgumentParser()

    #parser.add_argument("--train", type=strToBool, default=True, help="Train(True) or Demo(False)")
    parser.add_argument("--valid", type=bool, default=False, help="do validation(true) or false (bug: don'proxy.json set true)")
    parser.add_argument("--seed", type=int, default=None, help="fixed seed for random generator")
    parser.add_argument("--using_expression", type=bool, default=False, help="")
    parser.add_argument("--using_albedo_as_tex", type=bool, default=False, help="")

    # common
    parser.add_argument("--config_json", type=str, default=None, help="")

    parser.add_argument("--name", type=str, default="pretrain", help="running name")

    parser.add_argument("--checkpoint_root_path", type=str, default="./checkpoint", help="")
    parser.add_argument("--checkpoint_name", type=str, default=None,
                        help="if you using --checkpoint_name flag, --checkpoint_regex flag will be ignored")
    parser.add_argument("--checkpoint_regex", type=str, default=None, help="seconds")
    parser.add_argument("--checkpoint_epoch", type=int, default=None, help="")
    parser.add_argument("--checkpoint_step", type=int, default=None, help="")

    parser.add_argument("--dataset_path", type=str, default="./dataset", help="")
    parser.add_argument("--definition_path", type=str, default="./dataset/definition", help="")
    parser.add_argument("--prediction_src_path", type=str, default="pred/dst", help="")
    parser.add_argument("--prediction_dst_path", type=str, default="pred/src", help="")

    # Train policy
    parser.add_argument("--epoch", type=int, default=50, help="the number of training epochs")
    parser.add_argument("--batch_size", type=int, default=5, help="input batch size")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate for [[Generator]]")
    parser.add_argument("--betas", nargs=2, type=float, default=[0.5, 0.999], help="")
    parser.add_argument("--save_ratio", type=float, default=0.1, help="")

    # System environment
    parser.add_argument("--ngpu", type=int, default=1, help="the number of gpus")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="")
    parser.add_argument("--worker", type=int, default=1, help="the number of cpu workers for dataloader")

    # dataset
    parser.add_argument("--train_dataset_frac", type=float, default=1.0, help="")
    parser.add_argument("--valid_dataset_frac", type=float, default=1.0, help="")
    parser.add_argument("--test_dataset_frac", type=float, default=1.0, help="")

    # log
    parser.add_argument("--verbose", type=str, default="debug", choices=["debug", "info"], help="")
    parser.add_argument("--log_path", type=str, default="./logs", help="")
    parser.add_argument("--log_loss_interval", type=int, default=10, help="")
    parser.add_argument("--log_image_interval", type=int, default=50, help="")
    parser.add_argument("--log_image_count", type=int, default=4, help="")

    # 3dmm parameters
    parser.add_argument("--vertex_num", type=int, default=53215, help="")
    parser.add_argument("--tri_num", type=int, default=105840, help="")
    parser.add_argument("--vertex_num_reduce", type=int, default=39111, help="")
    parser.add_argument("--tri_num_reduce", type=int, default=77572, help="")
    parser.add_argument("--image_size", type=int, default=224, help="")
    parser.add_argument("--texture_size", nargs=2, type=int, default=[192, 224], help="")
    parser.add_argument("--landmark_num", type=int, default=68, help="")
    parser.add_argument("--c_dim", type=int, default=3, help="")
    parser.add_argument("--const_pixels_num", type=int, default=20, help="")

    # loss
    parser.add_argument("--start_loss_decay_step", type=int, default=0, help="")

    # supervised learning - common loss
    parser.add_argument("--m_loss", type=float, default=None, help="")
    parser.add_argument("--shape_loss", type=float, default=None, help="")
    parser.add_argument("--expression_loss", type=float, default=None, help="")
    parser.add_argument("--base_landmark_loss", type=float, default=None, help="")
    parser.add_argument("--comb_landmark_loss", type=float, default=None, help="")
    parser.add_argument("--gt_landmark_loss", type=float, default=None, help="")

    parser.add_argument("--m_loss_type", type=str, default="l2", choices=LOSS_TYPE_LIST, help="")
    parser.add_argument("--shape_loss_type", type=str, default="l2", choices=LOSS_TYPE_LIST, help="")
    parser.add_argument("--expression_loss_type", type=str, default="l2", choices=LOSS_TYPE_LIST, help="")
    parser.add_argument("--landmark_loss_type", type=str, default="l2", choices=LOSS_TYPE_LIST, help="")

    # supervised learning - texture loss
    parser.add_argument("--base_texture_loss", type=float, default=None, help="")
    parser.add_argument("--mix_ac_sb_texture_loss", type=float, default=None, help="")
    parser.add_argument("--mix_ab_sc_texture_loss", type=float, default=None, help="")
    parser.add_argument("--comb_texture_loss", type=float, default=None, help="")

    parser.add_argument("--texture_loss_type", type=str, default="l1", choices=LOSS_TYPE_LIST, help="")

    # perceptual loss
    parser.add_argument("--perceptual_layer", nargs="+", type=str,
                        default=["conv2d_1a", "conv2d_2a", "maxpool_3a", "conv2d_4a"], help="")
    parser.add_argument("--base_perceptual_recon_loss", type=float, default=None, help="")
    parser.add_argument("--mix_ab_sc_perceptual_recon_loss", type=float, default=None, help="")
    parser.add_argument("--mix_ac_sb_perceptual_recon_loss", type=float, default=None, help="")
    parser.add_argument("--comb_perceptual_recon_loss", type=float, default=None, help="")

    parser.add_argument("--perceptual_recon_loss_type", type=str, default="l2", choices=LOSS_TYPE_LIST, help="")

    # pixel loss
    parser.add_argument("--base_pix_recon_loss", type=float, default=None, help="")
    parser.add_argument("--mix_ab_sc_pix_recon_loss", type=float, default=None, help="")
    parser.add_argument("--mix_ac_sb_pix_recon_loss", type=float, default=None, help="")
    parser.add_argument("--comb_pix_recon_loss", type=float, default=None, help="")

    parser.add_argument("--pix_recon_loss_type", type=str, default="l1", choices=LOSS_TYPE_LIST, help="")

    # regularization
    parser.add_argument("--batchwise_white_shading_loss", type=float, default=None, help="")
    parser.add_argument("--symmetry_loss", type=float, default=None, help="")
    parser.add_argument("--const_albedo_loss", type=float, default=None, help="")
    parser.add_argument("--const_local_albedo_loss", type=float, default=None, help="")
    parser.add_argument("--shade_mag_loss", type=float, default=None, help="")

    parser.add_argument("--base_smoothness_loss", type=float, default=None, help="")
    parser.add_argument("--comb_smoothness_loss", type=float, default=None, help="")
    parser.add_argument("--base_exp_smoothness_loss", type=float, default=None, help="")
    parser.add_argument("--comb_exp_smoothness_loss", type=float, default=None, help="")

    parser.add_argument("--smoothness_loss_type", type=str, default="l2", choices=LOSS_TYPE_LIST, help="")

    parser.add_argument("--shape_residual_loss", type=float, default=None, help="")
    parser.add_argument("--albedo_residual_loss", type=float, default=None, help="")

    parser.add_argument("--batchwise_white_shading_loss_type", type=str, default="l2", choices=LOSS_TYPE_LIST, help="")
    parser.add_argument("--symmetry_loss_type", type=str, default="l1", choices=LOSS_TYPE_LIST, help="")
    parser.add_argument("--const_albedo_loss_type", type=str, default="l1", choices=LOSS_TYPE_LIST, help="")
    parser.add_argument("--const_local_albedo_loss_type", type=str, default="l2,1", choices=LOSS_TYPE_LIST, help="")
    parser.add_argument("--shade_mag_loss_type", type=str, default="l1", choices=LOSS_TYPE_LIST, help="")
    parser.add_argument("--residual_loss_type", type=str, default="l1", choices=LOSS_TYPE_LIST, help="")

    # random parameter
    parser.add_argument("--random_camera", action='store_true', default=False, help="")
    parser.add_argument("--random_expression", action='store_true', default=False, help="")
    parser.add_argument("--random_illumination", action='store_true', default=False, help="")
    parser.add_argument("--random_batch_sample_cnt", type=int, default=10, help="")  # batch * 10

    # deprecated
    # parser.add_argument("--identity_loss", type=float, default=None, help="")
    # parser.add_argument("--content_loss", type=float, default=None, help="")
    # parser.add_argument("--gradient_difference_loss", type=float, default=None, help="")

    args = parser.parse_args()
    sys_argv = sys.argv
    if args.config_json is not None:
        config_dict = json.load(open(args.config_json, "r"))
        for k, v in config_dict.items():
            if f"--{k}" not in sys_argv:
                setattr(args, k, v)

    losses = dict(filter(lambda pair: pair[0].endswith("loss") and pair[1] is not None, args.__dict__.items()))
    losses = {k.replace("_loss", ""): v for k, v in losses.items()}

    dels = []
    for k, v in args.__dict__.items():
        if k.endswith("loss"):
            dels.append(k)

    for k in dels:
        delattr(args, k)

    # 3dmm parameter
    setattr(args, "N", args.vertex_num * 3)
    setattr(args, "N_reduce", args.vertex_num_reduce * 3)

    # random
    setattr(args, "random_sample_num", args.batch_size * args.random_batch_sample_cnt)

    if args.seed is None:
        args.seed = random.randint(0, 1 << 30)
    random.seed(args.seed)
    torch.manual_seed(args.seed)


    if args.dtype == "float":
        setattr(args, "dtype", torch.float32)
    elif args.dtype == "double":
        setattr(args, "dtype", torch.double)
    else:
        raise(f"No such dtype : {args.dtype}")

    print("--- training settings ---\n\n")
    for k, v in args.__dict__.items():
        print(k, ":", v)
    print("--- training setting finish ---\n\n")

    

    return args, losses


def init_3dmm_settings():
    deep_to_blender = np.load(join(CFG.definition_path, 'deep_to_blender.npy'))
    blender_to_deep = np.load(join(CFG.definition_path, 'blender_to_deep.npy'))
    
    # mean_shape = np.load(join(CFG.definition_path, 'mean_shape.npy')).reshape([-1, 3])[blender_to_deep]
    mean_shape = np.load(join(CFG.definition_path, 'mean_shape.npy')).reshape([-1, 3])
    shapeBase = np.load(join(CFG.definition_path, 'shapeBase.npy'))
    shapeBase_inverse = np.matmul(np.linalg.inv(np.matmul(shapeBase.transpose(), shapeBase)), shapeBase.transpose()).transpose()

    exBase = np.load(join(CFG.definition_path, 'exBase.npy'))
    exBase_inverse = np.matmul(np.linalg.inv(np.matmul(exBase.transpose(), exBase)), exBase.transpose()).transpose()

    # mean_tex = np.reshape(np.load(join(CFG.definition_path, 'mean_tex.npy')) / 255.0, [-1, 3])[blender_to_deep]
    mean_tex = np.reshape(np.load(join(CFG.definition_path, 'mean_tex.npy')) / 255.0, [-1, 3])
    # mean_tex = mean_tex[:,::-1].copy()
    texBase = np.load(join(CFG.definition_path, 'texBase.npy'))
    texBase_inverse = np.matmul(np.linalg.inv(np.matmul(texBase.transpose(), texBase)), texBase.transpose()).transpose()
    

    h, w = CFG.texture_size
    vt2pixel_u, vt2pixel_v = torch.split(torch.tensor(np.load(join(CFG.definition_path, 'BFM_uvmap.npy')), dtype=CFG.dtype), (1, 1), dim=-1)
    vt2pixel_v = torch.ones_like(vt2pixel_v) - vt2pixel_v
    vt2pixel_u, vt2pixel_v = vt2pixel_u * h, vt2pixel_v * w
    vt2pixel_u = vt2pixel_u[blender_to_deep]
    vt2pixel_v = vt2pixel_v[blender_to_deep]
    # vt2pixel_u, vt2pixel_v = vt2pixel_v, vt2pixel_u
    
    landmark = np.load(join(CFG.definition_path, 'landmark.npy')) - 1
    # landmark = deep_to_blender[landmark]


    face = np.load(join(CFG.definition_path, 'face.npy')) - 1
    # face = deep_to_blender[face]
    
    point_buf = np.load(join(CFG.definition_path, 'point_buf.npy')) - 1
    # point_buf = point_buf[blender_to_deep]

    const_alb_mask = 255 - np.load(join(CFG.definition_path, 'const_alb_mask.npy'))

    
    

    global_3dmm_setting = dict(
        mean_shape=torch.tensor(mean_shape, dtype=CFG.dtype).to(CFG.device),
        mean_shape_cpu=torch.tensor(mean_shape, dtype=CFG.dtype),
        shapeBase=torch.tensor(shapeBase, dtype=CFG.dtype).to(CFG.device),
        shapeBase_cpu=torch.tensor(shapeBase, dtype=CFG.dtype),
        shapeBase_inverse=torch.tensor(shapeBase_inverse, dtype=CFG.dtype).to(CFG.device),
        shapeBase_inverse_cpu=torch.tensor(shapeBase_inverse, dtype=CFG.dtype),
        exBase=torch.tensor(exBase, dtype=CFG.dtype).to(CFG.device),
        exBase_cpu=torch.tensor(exBase, dtype=CFG.dtype),
        exBase_inverse=torch.tensor(exBase_inverse, dtype=CFG.dtype).to(CFG.device),
        exBase_inverse_cpu=torch.tensor(exBase_inverse, dtype=CFG.dtype),
        mean_tex=torch.tensor(mean_tex, dtype=CFG.dtype).to(CFG.device),
        mean_tex_cpu=torch.tensor(mean_tex, dtype=CFG.dtype),
        texBase_inverse=torch.tensor(texBase_inverse, dtype=CFG.dtype).to(CFG.device),
        texBase_inverse_cpu=torch.tensor(texBase_inverse, dtype=CFG.dtype),
        texBase=torch.tensor(texBase, dtype=CFG.dtype).to(CFG.device),
        texBase_cpu=torch.tensor(texBase, dtype=CFG.dtype),

        face_cpu=torch.tensor(face, dtype=torch.int32).contiguous(),
        face=torch.tensor(face, dtype=torch.int32).contiguous().to(CFG.device),
    
        vt2pixel_u=vt2pixel_u.to(CFG.device),
        vt2pixel_v=vt2pixel_v.to(CFG.device),
        
        landmark=torch.tensor(landmark, dtype=torch.int64).to(CFG.device),

        point_buf_cpu=torch.tensor(point_buf, dtype=torch.int32),
        point_buf=torch.tensor(point_buf, dtype=torch.int32).to(CFG.device),
        const_alb_mask=torch.tensor(const_alb_mask, dtype=torch.int32),

        deep_to_blender_cpu=torch.tensor(deep_to_blender),
        deep_to_blender=torch.tensor(deep_to_blender).to(CFG.device),

        blender_to_deep_cpu=torch.tensor(blender_to_deep),
        blender_to_deep=torch.tensor(blender_to_deep).to(CFG.device),
        

        # mean_m=torch.tensor(np.load(join(CFG.dataset_path, 'mean_m.npy')), dtype=torch.float32).to(CFG.device),
        # std_m=torch.tensor(np.load(join(CFG.dataset_path, 'std_m.npy')), dtype=torch.float32).to(CFG.device),
        #
        # mean_exp=torch.tensor(mu_exp, dtype=torch.float32).to(CFG.device),
        # std_exp=torch.tensor(np.tile(np.array([1e4, 1e4, 1e4]), CFG.vertex_num), dtype=torch.float32).to(CFG.device),
        #
        # w_shape=torch.tensor(w_shape, dtype=torch.float32).to(CFG.device),
        # w_exp=torch.tensor(w_exp, dtype=torch.float32).to(CFG.device),
        #
        # tri=tri,
        # face=face,
    )

    for key, value in global_3dmm_setting.items():
        setattr(CFG, key, value)



CFG, LOSSES = parse()
