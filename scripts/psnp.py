import os
from argparse import Namespace

from tqdm import tqdm
import time
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
import sys
import faiss

sys.path.append(".")
sys.path.append("..")

from configs import data_configs
from datasets.inference_dataset import InferenceDatasetWithPath
from utils.common import tensor2im, log_input_image
from options.test_options import TestOptions
from models.psp import pSp


def main():

    # path setup
    test_opts = TestOptions().parse()
    out_path_results = os.path.join(test_opts.exp_dir, 'inference_results')
    out_path_latents = os.path.join(test_opts.exp_dir, 'inference_latents')
    out_path_coupled = os.path.join(test_opts.exp_dir, 'inference_coupled')
    os.makedirs(out_path_results, exist_ok=True)
    os.makedirs(out_path_coupled, exist_ok=True)
    os.makedirs(out_path_latents, exist_ok=True)

    # update test options with options used during training
    ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
    opts = ckpt['opts']
    opts.update(vars(test_opts))
    if 'learn_in_w' not in opts:
        opts['learn_in_w'] = False
    opts = Namespace(**opts)

    # model setup
    net = pSp(opts)
    net.eval()
    net.cuda()

    # dataset setup
    print('Loading dataset for {}'.format(opts.dataset_type))
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    dataset = InferenceDatasetWithPath(
        root=opts.data_path,
        transform=transforms_dict['transform_inference'],
        opts=opts
    )
    dataloader = DataLoader(
        dataset,
        batch_size=opts.test_batch_size,
        shuffle=True,
        num_workers=int(opts.test_workers),
        drop_last=True
    )

    # n images to generate
    if opts.n_images is None:
        opts.n_images = len(dataset)

    # inference setup
    global_i = 0
    global_time = []

    # inference
    for input_batch, input_paths in tqdm(dataloader):

        if global_i >= opts.n_images:
            break

        with torch.no_grad():

            input_cuda = input_batch.cuda().float()
            tic = time.time()

            result_batch, result_latents = run_on_batch(input_cuda, net, opts)

            latent_array = result_latents.cpu().detach().numpy()
            latents_save_path = os.path.join(out_path_latents, f'{global_i}.npy')
            with open(latents_save_path, 'wb') as f:
                np.save(f, latent_array)

            toc = time.time()
            global_time.append(toc - tic)

        for i in range(opts.test_batch_size):

            result = tensor2im(result_batch[i])
            im_path = input_paths[i]

            if opts.couple_outputs:

                input_im = log_input_image(input_batch[i], opts)
                resize_amount = (256, 256) if opts.resize_outputs else (opts.output_size, opts.output_size)
                res = np.concatenate(
                    [np.array(input_im.resize(resize_amount)),
                    np.array(result.resize(resize_amount))], axis=1)
                Image.fromarray(res).save(os.path.join(out_path_coupled, os.path.basename(im_path)))

            im_save_path = os.path.join(out_path_results, os.path.basename(im_path))
            Image.fromarray(np.array(result)).save(im_save_path)

            global_i += 1

    # create stats
    stats_path = os.path.join(opts.exp_dir, 'stats.txt')
    result_str = 'Runtime {:.4f}+-{:.4f}'.format(np.mean(global_time), np.std(global_time))
    print(result_str)
    with open(stats_path, 'w') as f:
        f.write(result_str)

    # latents similarity search
    if opts.faiss_dir is not None:

        # create index
        FIRST_N_LATENTS = 2
        DIM = 1024
        index = faiss.IndexFlatL2(DIM)
        print(index.is_trained)

        # load index
        for root, dirs, files in os.walk(opts.faiss_dir):
            for name in files:
                if name.endswith('.npy'):
                    with open(os.path.join(root, name), 'rb') as f:
                        saved_latents = np.load(f)
                        reshaped_latents = saved_latents[:,:FIRST_N_LATENTS,:].\
                            reshape((saved_latents.shape[0], -1))
                        assert reshaped_latents.shape[1] == DIM
                        index.add(reshaped_latents)

        print(index.ntotal)

        # search index
        k = 4                                    # we want to see 4 nearest neighbors
        D, I = index.search(reshaped_latents, k) # sanity check
        print(I)
        print(D)          


def run_on_batch(inputs, net, opts):

    # No style mixing inference
    if opts.latent_mask is None:

        result_batch, result_latents = net(
            inputs, 
            randomize_noise=False, 
            resize=opts.resize_outputs, 
            return_latents=True
        )

    else:
        raise NotImplementedError

    return result_batch, result_latents


if __name__ == '__main__':
    main()
