import os
import math
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from gptnano.translator import VQGANTransformer
from utils import load_data, plot_images
import matplotlib.pyplot as plt
from gptnano.nanogpt import LayerNorm
from contextlib import nullcontext
import inspect


# given a numpy array of images, plot them in a grid with n rows and m columns
def save_images(images, rows, cols, path):
    fig, axes = plt.subplots(
        rows, cols, figsize=(20, 20))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(images[i], cmap='gray')
        ax.axis('off')
    plt.savefig(path)


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


class TrainTransformer:
    def __init__(self, args):
        self.model = VQGANTransformer(args).to(device=args.device)
        self.optim = self.configure_optimizers()

        self.train(args)

    def configure_optimizers(self):
        device_type = 'cuda'
        weight_decay = 0.01
        self.learning_rate = 6e-4  # 4.5e-06
        betas = (0.9, 0.95)

        optimizer = self.model.transformer.configure_optimizers(
            weight_decay=weight_decay,
            learning_rate=self.learning_rate,
            betas=betas,
            device_type=device_type,
        )

        return optimizer

    def setup_training(self, args):
        seed_offset = 0
        device = 'cuda'
        # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
        dtype = 'bfloat16'
        compile = True
        grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0

        torch.manual_seed(1337 + seed_offset)
        torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
        # for later use in torch.autocast
        device_type = 'cuda' if 'cuda' in device else 'cpu'
        # note: float16 data type will automatically use a GradScaler
        ptdtype = {'float32': torch.float32,
                   'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
        ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(
            device_type=device_type, dtype=ptdtype)

        # initialize a GradScaler. If enabled=False scaler is a no-op
        scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

        # compile the model
        if compile:
            print("compiling the model... (takes a ~minute)")
            unoptimized_model = self.model
            self.model = torch.compile(
                unoptimized_model)  # requires PyTorch 2.0

        return ctx, scaler, grad_clip

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(self, it):
        warmup_iters = 2000
        learning_rate = self.learning_rate
        lr_decay_iters = 600000
        min_lr = 6e-5

        # 1) linear warmup for warmup_iters steps
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > lr_decay_iters:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        # coeff ranges 0..1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (learning_rate - min_lr)

    def train(self, args):
        decay_lr = True
        ctx, scaler, grad_clip = self.setup_training(args)

        train_dataloader, val_dataloader, test_dataloader = load_data(args)
        current_best_val_epoch_loss = torch.tensor(float('inf'))
        current_best_epoch = 0
        run_idx = 0
        while os.path.exists(os.path.join("gpt_results", f"run_{run_idx}")):
            run_idx += 1

        iter_num = 0
        for epoch in range(args.epochs):
            with tqdm(range(len(train_dataloader))) as pbar:
                train_epoch_loss = []
                for i, data in zip(pbar, train_dataloader):
                    # determine and set the learning rate for this iteration
                    lr = self.get_lr(
                        iter_num) if decay_lr else self.learning_rate
                    for param_group in self.optim.param_groups:
                        param_group['lr'] = lr

                    imgs_ct = data['indices_ct']
                    imgs_ap = data['indices_ap']
                    imgs_lat = data['indices_lat']

                    self.optim.zero_grad()

                    imgs_ct = imgs_ct.to(device=args.device)
                    imgs_ap = imgs_ap.to(device=args.device)
                    imgs_lat = imgs_lat.to(device=args.device)

                    with ctx:
                        logits, targets = self.model(
                            (imgs_ct, imgs_ap, imgs_lat))

                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

                    # loss.backward()
                    # backward pass, with gradient scaling if training in fp16
                    scaler.scale(loss).backward()

                    # clip the gradient
                    if grad_clip != 0.0:
                        scaler.unscale_(self.optim)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), grad_clip)
                    # step the optimizer and scaler if training in fp16
                    scaler.step(self.optim)
                    scaler.update()

                    # self.optim.step()
                    pbar.set_postfix(Transformer_Loss=np.round(
                        loss.item(), 4), lr=np.round(lr, 6))
                    pbar.update(0)
                    train_epoch_loss.append(loss)

                    iter_num += 1
                print("Epoch: ", epoch, "Loss (mean): ",
                      torch.mean(torch.tensor(train_epoch_loss)))

            with tqdm(range(len(val_dataloader))) as pbar:
                val_epoch_loss = []
                self.model.eval()
                with torch.no_grad():
                    for i, data in zip(pbar, val_dataloader):
                        imgs_ct = data['indices_ct']
                        imgs_ap = data['indices_ap']
                        imgs_lat = data['indices_lat']

                        imgs_ct = imgs_ct.to(device=args.device)
                        imgs_ap = imgs_ap.to(device=args.device)
                        imgs_lat = imgs_lat.to(device=args.device)

                        with ctx:
                            logits, targets = self.model(
                                (imgs_ct, imgs_ap, imgs_lat))

                        loss = F.cross_entropy(
                            logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                        pbar.set_postfix(Transformer_Loss=np.round(
                            loss.item(), 4))
                        pbar.update(0)
                        val_epoch_loss.append(loss)
                mean_val_epoch_loss = torch.mean(torch.tensor(val_epoch_loss))
                print("Epoch: ", epoch, "Loss (mean): ", mean_val_epoch_loss)
                self.model.train()


            checkpoint_dir = os.path.join(
                    "gpt_results", f"run_{run_idx}", "checkpoints")
            create_dir(checkpoint_dir)

            # Store last checkpoint
            torch.save(self.model.state_dict(), os.path.join(
                checkpoint_dir, f"transformer_last.pt"))

			# Store best checkpoint
            if mean_val_epoch_loss < current_best_val_epoch_loss:
                torch.save(self.model.state_dict(), os.path.join(
                    checkpoint_dir, f"transformer_{epoch}_{np.round(mean_val_epoch_loss.float(), 4)}.pt"))

                # check if file exists
                path = os.path.join(
                    checkpoint_dir, f"transformer_{current_best_epoch}_{np.round(current_best_val_epoch_loss.float(), 4)}.pt")
                if os.path.isfile(path):
                    os.remove(path)

                current_best_val_epoch_loss = mean_val_epoch_loss
                current_best_epoch = epoch

            if epoch % 10 == 0:
                self.model.eval()
                with ctx:
                    log, sampled_imgs_ct, sampled_imgs_ap, sampled_imgs_lat = self.model.log_images(
                        (imgs_ct[0][None], imgs_ap[0][None], imgs_lat[0][None]))
                SLICE_IDX = 60

                image_dir = os.path.join(
                    f"gpt_results", f"run_{run_idx}", "results")

                create_dir(image_dir)

                np_imgs = torch.cat((sampled_imgs_ct[:, :, :, SLICE_IDX].flip(
                    dims=(1, 2)), sampled_imgs_ap, sampled_imgs_lat)).detach().cpu().to(torch.float).numpy()[:, 0]
                save_images(np_imgs, rows=3, cols=3, path=os.path.join(
                    image_dir, f"transformer_{epoch}.jpg"))

                # vutils.save_image(torch.cat((sampled_imgs_ct[:, :, :, SLICE_IDX].flip(dims=(1, 2)), sampled_imgs_ap, sampled_imgs_lat)), os.path.join(
                # image_dir, f"transformer_{epoch}.jpg"), nrow=3)
                # plot_images(log)

                self.model.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN")
    # VQGAN
    parser.add_argument('--num-codebook-vectors', type=int,
                        default=8192, help='Number of codebook vectors.')
    parser.add_argument('--checkpoint-path-3d-vqgan', type=str,
                        default=None, help='Path to checkpoint.')

    parser.add_argument('--checkpoint-path-2d-vqgan', type=str,
                        default=None, help='Path to checkpoint.')

    parser.add_argument('path-to-preprocessed-data', type=str, default=None, help='Path to preprocessed LIDC-IDRI data')
    parser.add_arguments('path-to-data-indices', type=str, default=None, help='Path to pre-extracted indices')

    # Transformer
    parser.add_argument('--pkeep', type=float, default=0.5,
                        help='Percentage for how much latent codes to keep.')
    parser.add_argument('--sos-token', type=int, default=0,
                        help='Start of Sentence token.')
    # TODO: Provide better explanation and maybe also why 4096 + 256 * 2 (because of ct, lat and ap)
    parser.add_argument('--block_size', type=int, default=4096 + 256 * 2,
                        help='Block size of GPT')
    # TODO: Provide better explanation and maybe also why 256 * 2 (because of lat and ap) and +1 to include the sos token
    parser.add_argument('--n_unmasked', type=int, default=256 * 2 + 1,
                        help='Number of unmasked tokens (needed for the 2D images)')
    parser.add_argument('--use_normal_attention',
                        action='store_true', help='Use normal attention instead of performer attention.')

    # Training
    parser.add_argument('--device', type=str, default="cuda",
                        help='Which device the training is on')
    parser.add_argument('--batch-size', type=int, default=12,
                        help='Input batch size for training.')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of epochs to train.')
    parser.add_argument('--learning-rate', type=float,
                        default=2.25e-05, help='Learning rate.')
    parser.add_argument('--num_workers', type=int,
                        default=20, help='Number of workers for the dataloader.')

    args = parser.parse_args()
    # create dir if not exists
    train_transformer = TrainTransformer(args)


# TODO: Add Performer
# TODO: Add GAN in latent space
