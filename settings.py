import argparse
import sys
import json


LOSS_TYPE_LIST = ["l1", "l2", "l2,1"]

def parse():
    parser = argparse.ArgumentParser()

    #parser.add_argument("--train", type=strToBool, default=True, help="Train(True) or Demo(False)")
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
    parser.add_argument("--definition_path", type=str, default="./dataset/3DMM_definition/", help="")
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
    parser.add_argument("--shape_residual_loss", type=float, default=None, help="")
    parser.add_argument("--albedo_residual_loss", type=float, default=None, help="")

    parser.add_argument("--batchwise_white_shading_loss_type", type=str, default="l2", choices=LOSS_TYPE_LIST, help="")
    parser.add_argument("--symmetry_loss_type", type=str, default="l1", choices=LOSS_TYPE_LIST, help="")
    parser.add_argument("--smoothness_loss_type", type=str, default="l2", choices=LOSS_TYPE_LIST, help="")
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

    print("--- training settings ---\n\n")
    for k, v in args.__dict__.items():
        print(k, ":", v)
    print("--- training setting finish ---\n\n")
    return args, losses


CFG, LOSSES = parse()
