import os
import argparse
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob
import numpy as np

from ddcm.latent_runners import get_loader, compress
from ddcm.latent_models import load_model
from ddcm.util.img_utils import clear_color
from ddcm.util.file import get_args_from_filename, save_as_binary_bitwise, load_from_binary_bitwise


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True, help="Mode of operation")

    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument('--gpu', type=int, default=0, help='GPU device index to use')
    common_parser.add_argument('--float16', action='store_true', help='Use float16 precision for model inference')
    common_parser.add_argument('--input_dir', required=True, help="Directory containing images to compress or bin files to decmopress.")
    common_parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the results')

    compression_args_parser = argparse.ArgumentParser(add_help=False)
    compression_args_parser.add_argument('--model_id', type=str, required=True, help='Pre-trained diffusion model to use',
                                         choices=['CompVis/stable-diffusion-v1-4',
                                                  'stabilityai/stable-diffusion-2-1',
                                                  'stabilityai/stable-diffusion-2-1-base'
                                                  ])
    compression_args_parser.add_argument('-K', '--num_noises', dest='K', type=int, default=1024, help="Codebook size")
    compression_args_parser.add_argument('-T', '--timesteps', dest='T', type=int, default=1000, help='Compress using T diffusion steps.')
    compression_args_parser.add_argument('-M', '--num_pursuit_noises', dest='M', type=int, default=1, help="Atoms in the MP version. MP starts when M > 1.")
    compression_args_parser.add_argument('-C', '--num_pursuit_coef_bits', dest='C', type=int, default=1, help="Amount of discrete coefficients for MP.")
    compression_args_parser.add_argument('--t_range', nargs=2, type=int, default=[999, 0], help="Optimize only a subset of the timesteps range.")

    # latent_compression.py  -> main() 里 parser 定义之后
    ### >>> ADD（放在创建 subparsers 之后、parser.parse_args() 之前）
    common_parser.add_argument('--wm_kind',
                        type=str,
                        default='none',  # 'none' | 'chacha' | 'xor'
                        choices=['none', 'chacha', 'xor'],
                        help='Watermarked initial-noise kind. Use "chacha" or "xor" to enable; "none" to disable.')

    compress_parser = subparsers.add_parser('compress', help='Compress a file', parents=[common_parser, compression_args_parser])
    decompress_parser = subparsers.add_parser('decompress', help='Decompress a file', parents=[common_parser])
    roundtrip_parser = subparsers.add_parser('roundtrip', help='Compress and then decompress', parents=[common_parser, compression_args_parser])

    args = parser.parse_args()

    # 在 args = parser.parse_args() 之后，紧接着：
    ### >>> ADD（固定 watermark 参数；你不改这几行就永远生成同一个 w）
    WM_KIND = args.wm_kind
    WM_ENABLE = (WM_KIND != 'none')
    WM_SEED = 123456        # << 写死
    WM_CH = 1               # << 写死
    WM_HW = 8               # << 写死 (=> 256-bit watermark)
    WM_FPR = 1e-6           # << 写死
    WM_USER_NUMBER = 1      # << 写死

    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)

    if args.mode == 'decompress':
        # get the first indices file in the input directory
        indices_file = glob(os.path.join(args.input_dir, '**', '*.bin'), recursive=True)[0]
        args.T, args.K, args.M, args.C, args.t_range, args.model_id = get_args_from_filename(indices_file)

    # Load model
    model, _ = load_model(args.model_id, args.T, device, args.float16, compile=False)
    imsize = model.get_image_size()

    optimized_Ts = len([t for t in model.timesteps if (args.t_range[0] >= t >= args.t_range[1])])
    if args.mode == 'compress' or args.mode == 'roundtrip':
        out_prefix = f"T={args.T}_in{args.t_range[0]}-{args.t_range[1]}_K={args.K}_M={args.M}_C={args.C}_model={args.model_id.split('/')[1]}"
        bpp = (optimized_Ts - 1) * (args.M * np.log2(args.K) + (args.M - 1) * args.C) / (imsize ** 2)
        print(f'The BPP will be: {bpp:.4f}')
    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == 'compress' or args.mode == 'roundtrip':
        loader = get_loader(args.input_dir, resize_to=imsize)

        # Do Inference
        for orig_img, orig_path in loader:
            respath = os.path.join(args.output_dir, out_prefix, os.path.dirname(orig_path[0]))
            os.makedirs(respath, exist_ok=True)  # keep dir structure
            fname = os.path.basename(orig_path[0]).split(os.extsep)[0]

            orig_img = orig_img.to(device)
            ### >>> REPLACE（解四个返回；把 wm_* 常量传下去；保存 wm_meta）
            with torch.no_grad():
                compressed_im, noise_indices, coeff_indices, wm_meta = compress(
                    model, orig_img, args.K, device, args.M, args.C, args.t_range,
                    # 下面是 watermark 开关与固定参数
                    wm_enable=WM_ENABLE,
                    wm_kind=WM_KIND,
                    wm_seed=WM_SEED,
                    wm_ch=WM_CH,
                    wm_hw=WM_HW,
                    wm_fpr=WM_FPR,
                    wm_user_number=WM_USER_NUMBER
                )

            plt.imsave(os.path.join(respath, f'{fname}_comp.png'), clear_color(compressed_im, normalize=False))

            # 保存索引（原有）
            save_as_binary_bitwise(noise_indices.numpy(), coeff_indices.numpy(),
                                args.K, args.M, args.C,
                                os.path.join(respath, f'{fname}_noise_indices.bin'))

            # 仅在压缩/roundtrip & 启用水印时，保存“用于验证”的元数据（与 bin 同目录）
            if WM_ENABLE and wm_meta is not None:
                wm_meta_path = os.path.join(respath, f'{fname}_wm_meta.pt')
                # 如果你只想“第一次保存”，避免覆盖可加 exists 判断：
                if not os.path.exists(wm_meta_path):
                    torch.save(wm_meta, wm_meta_path)


    if args.mode == 'roundtrip':
        args.input_dir = os.path.join(args.output_dir, out_prefix)
        args.output_dir = args.input_dir

    if args.mode == 'decompress' or args.mode == 'roundtrip':
        indices_files = glob(os.path.join(args.input_dir, '**', '*.bin'), recursive=True)
        for indices_file in tqdm(indices_files, desc="Decompressing files"):
            indices = load_from_binary_bitwise(indices_file, args.K, args.M, args.C, optimized_Ts)
            respath = os.path.join(args.output_dir, Path(os.path.dirname(indices_file)).relative_to(args.input_dir))
            os.makedirs(respath, exist_ok=True)
            fname = os.path.basename(indices_file).split(os.extsep)[0].split('_noise_indices')[0]

            ### >>> REPLACE（把 watermark 常量传下去；忽略 wm_meta）
            with torch.no_grad():
                decompressed_im, _, _, _ = compress(
                    model, None, args.K, device, args.M, args.C, args.t_range, indices,
                    wm_enable=WM_ENABLE,
                    wm_kind=WM_KIND,
                    wm_seed=WM_SEED,
                    wm_ch=WM_CH,
                    wm_hw=WM_HW,
                    wm_fpr=WM_FPR,
                    wm_user_number=WM_USER_NUMBER
                )

            plt.imsave(os.path.join(respath, f'{fname}_decomp.png'), clear_color(decompressed_im, normalize=False))


if __name__ == '__main__':
    main()
