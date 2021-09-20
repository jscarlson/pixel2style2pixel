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
from numpy.lib.format import open_memmap

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
    out_path_coupled = os.path.join(test_opts.exp_dir, 'inference_coupled')
    out_path_latents = os.path.join(test_opts.faiss_dir, 'inference_latents')
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
        num_workers=int(opts.test_workers)
    )

    # n images to generate
    if opts.n_images is None:
        opts.n_images = len(dataset)

    # inference setup
    global_i = 0
    global_time = []

    # read in latents
    if not opts.save_latents:
        index = faiss.read_index(os.path.join(opts.faiss_dir, 'index.bin'))
        lookup_arrays = np.load(os.path.join(opts.faiss_dir, 'lookup_array.npy'), mmap_mode='r')
        
    # inference
    for input_batch, input_paths in tqdm(dataloader):

        if global_i >= opts.n_images:
            break

        with torch.no_grad():

            input_cuda = input_batch.cuda().float()
            tic = time.time()

            result_batch, result_latents = run_on_batch(input_cuda, net, opts)

            if opts.save_latents:

                latent_array = result_latents.cpu().detach().numpy().astype('float32')
                latents_save_path = os.path.join(out_path_latents, f'{global_i}.npy')
                with open(latents_save_path, 'wb') as f:
                    np.save(f, latent_array)
                
            else:
                
                closest_latents = run_faiss(
                    result_latents, 
                    index, 
                    lookup_arrays, 
                    n_latents=opts.n_latents, 
                    n_neighbors=opts.n_neighbors
                )

                for bidx, clatent in enumerate(closest_latents):

                    closest_input_cuda = torch.from_numpy(clatent).cuda().float()
                    result_neighbors, _ = run_on_batch(closest_input_cuda, net, opts, input_code=True)

                    im_path = input_paths[bidx]
                    input_im = log_input_image(input_batch[bidx], opts)
                    res = [np.array(input_im)]
                    res = res + [np.array(tensor2im(result_neighbors[i])) for i in range(opts.n_neighbors)]
                    res = np.concatenate(res, axis=1)
                    Image.fromarray(res).save(os.path.join(out_path_coupled, os.path.basename(im_path)))

            toc = time.time()
            global_time.append(toc - tic)

        if opts.save_images and opts.save_latents:

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

        global_i += opts.test_batch_size

    # faiss index creation
    if opts.save_latents:
        index, lookup_arrays = setup_faiss(opts, n_latents=opts.n_latents)
        faiss.write_index(index, os.path.join(opts.faiss_dir, 'index.bin'))
        with open(os.path.join(opts.faiss_dir, 'lookup_array.npy'), 'wb') as f:
            np.save(f, lookup_arrays)
        
    # create stats
    stats_path = os.path.join(opts.exp_dir, 'stats.txt')
    result_str = 'Runtime {:.4f}+-{:.4f}'.format(np.mean(global_time), np.std(global_time))
    print(result_str)
    with open(stats_path, 'w') as f:
        f.write(result_str)


def setup_faiss(opts, n_latents, dim=512):

    # create index
    index = faiss.IndexFlatIP(dim*n_latents)
    all_arrays = np.empty((0, 10, dim), dtype=np.float32)

    # load index
    for root, dirs, files in os.walk(opts.faiss_dir):
        for name in tqdm(files, position=0, leave=True):
            if name.endswith('.npy'):
                saved_latents = np.load(os.path.join(root, name))
                all_arrays = np.concatenate([all_arrays, saved_latents], axis=0)
                reshaped_latents = reshape_latent(saved_latents, n_latents)
                faiss.normalize_L2(reshaped_latents)
                index.add(reshaped_latents)
    print(f'Total indices {index.ntotal}')

    return index, all_arrays


def run_faiss(query_latents, index, all_arrays, n_latents, n_neighbors=5, verbose=True):
    
    # search index
    reshaped_query_latents = reshape_latent(query_latents, n_latents)
    D, I = index.search(reshaped_query_latents, n_neighbors)
    if verbose:
        print(I)
        print(D)

    # return closest
    closest_indices = np.apply_along_axis(lambda x: x[:n_neighbors], axis=1, arr=I)
    closest_codes = [all_arrays[cidx,:,:] for cidx in closest_indices]

    return closest_codes


def reshape_latent(latents, n_latents):
    if torch.is_tensor(latents):
        latents = latents.cpu().detach().numpy()
    return np.ascontiguousarray(
        latents[:,:n_latents,:].reshape((latents.shape[0], -1))
    )


def run_on_batch(inputs, net, opts, input_code=False):

    # No style mixing inference
    if opts.latent_mask is None:

        result_batch, result_latents = net(
            inputs, 
            randomize_noise=False, 
            resize=opts.resize_outputs, 
            return_latents=True,
            input_code=input_code,
        )

    else:
        raise NotImplementedError

    return result_batch, result_latents


if __name__ == '__main__':
    main()
