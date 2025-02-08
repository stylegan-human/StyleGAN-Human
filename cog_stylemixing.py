import os
import random
import tempfile
import typing

import cog
import numpy as np
import PIL.Image
import torch

import dnnlib
import legacy

NETWORK_PKL = "./pretrained_models/stylegan2_1024.pkl"
NUM_ROWS=4
NUM_COLS=6
COL_STYLES = "0-6"

class Predictor(cog.BasePredictor):
    def setup(self):
        self.device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
        with dnnlib.util.open_url(NETWORK_PKL) as f:
            self.G = legacy.load_network_pkl(f)['G_ema'].to(self.device)

    def predict(self,
        row_seeds: str = cog.Input(description=f"Seeds for the source image style. Comma separated list of integers (max {NUM_ROWS}). e.g. `85,100,75,458,1500`. Leave empty to use random seeds.", default=""),
        col_seeds: str = cog.Input(description=f"Seeds for the destination image styles. Comma separated list of integers (max {NUM_COLS}). e.g. `821,1789,293`. Leave empty to use random seeds." , default=""),
        truncation_psi: float = cog.Input(description='Truncation psi', default=1.0, ge=0.4, le=1.0),
        noise_mode: str = cog.Input(description='Noise mode', default='const', choices=['const', 'random', 'none']),
    ) -> typing.Iterator[cog.Path]:

        outdir = cog.Path(tempfile.mkdtemp())
        os.makedirs(outdir, exist_ok=True)

        # Prepare seeds
        if len(col_seeds) == 0:
            col_seeds = [random.randint(0, 2**31 - 1) for _ in range(NUM_COLS)]
        else:
            col_seeds = legacy.num_range(col_seeds)[:NUM_COLS]
        if len(row_seeds) == 0:
            row_seeds = [random.randint(0, 2**31 - 1) for _ in range(NUM_ROWS)]
        else:
            row_seeds = legacy.num_range(row_seeds)[:NUM_ROWS]
        col_styles = legacy.num_range(COL_STYLES)
        max_style = int(2 * np.log2(self.G.img_resolution)) - 3
        assert max(col_styles) <= max_style, f"Maximum col-style allowed: {max_style}, got col_styles: {col_styles}"
        print(f'Row seeds: {row_seeds}')
        print(f"Col seeds: {col_seeds}")
        print(f"Col styles: {col_styles}")

        print('Generating W vectors...')
        all_seeds = list(set(row_seeds + col_seeds))
        all_z = np.stack([np.random.RandomState(seed).randn(self.G.z_dim) for seed in all_seeds])
        all_w = self.G.mapping(torch.from_numpy(all_z).to(self.device), None)
        w_avg = self.G.mapping.w_avg
        all_w = w_avg + (all_w - w_avg) * truncation_psi
        w_dict = {seed: w for seed, w in zip(all_seeds, list(all_w))}

        print('Generating images...')
        all_images = self.G.synthesis(all_w, noise_mode=noise_mode)
        all_images = (all_images.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
        image_dict = {(seed, seed): image for seed, image in zip(all_seeds, list(all_images))}

        print('Generating style-mixed images...')
        for row_seed in row_seeds:
            for col_seed in col_seeds:
                w = w_dict[row_seed].clone()
                w[col_styles] = w_dict[col_seed][col_styles]
                image = self.G.synthesis(w[np.newaxis], noise_mode=noise_mode)
                image = (image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                image_dict[(row_seed, col_seed)] = image[0].cpu().numpy()

        print('Saving image grid...')
        W = self.G.img_resolution // 2
        H = self.G.img_resolution 
        canvas = PIL.Image.new('RGB', (W * (len(col_seeds) + 1), H * (len(row_seeds) + 1)), 'black')
        for row_idx, row_seed in enumerate([0] + row_seeds):
            for col_idx, col_seed in enumerate([0] + col_seeds):
                if row_idx == 0 and col_idx == 0:
                    continue
                key = (row_seed, col_seed)
                if row_idx == 0:
                    key = (col_seed, col_seed)
                if col_idx == 0:
                    key = (row_seed, row_seed)
                canvas.paste(PIL.Image.fromarray(image_dict[key], 'RGB'), (W * col_idx, H * row_idx))
            canvas.save(outdir.joinpath(f'grid.png'))
        yield cog.Path(outdir.joinpath('grid.png'))