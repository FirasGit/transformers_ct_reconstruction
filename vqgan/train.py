"""
Adapted from https://github.com/mueller-franzes/medfusion
"""

from pathlib import Path
from datetime import datetime
import argparse

import torch
from torch.utils.data import ConcatDataset
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


from datasets.data_module import SimpleDataModule
from datasets import LIDCDataset
from vqgan.model import VQVAE, VQGAN, VAE, VAEGAN

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def parse_args():
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument("--mode", type=str, choices=["2D", "3D"], default="2D", help="Whether to train 2D or 3D model.")
    parser.add_argument("--data-dir", type=str, default="/data/LIDC/preprocessed", help="Directory containing preprocessed LIDC-IDRI data.")
    parser.add_arguments("--best-vq-vae-ckpt", type=str, default=None, help="Path to the best checkpoint for the VQ-VAE model.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # --------------- Settings --------------------
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    path_run_dir = Path.cwd() / 'runs' / str(current_time)
    path_run_dir.mkdir(parents=True, exist_ok=True)
    gpus = [0] if torch.cuda.is_available() else None

    if args.mode == "2D":
            lidc_dataset_train = LIDCDataset(
                root_dir=args.data_dir, augmentation=False, projection=True, split='train')

            lidc_dataset_val = LIDCDataset(
                root_dir=args.data_dir, augmentation=False, projection=True, split='val')

            lidc_dataset_test = LIDCDataset(
                root_dir=args.data_dir, augmentation=False, projection=True, split='test')

            dm = SimpleDataModule(
                ds_train=lidc_dataset_train,
                ds_val=lidc_dataset_val,
                ds_test=lidc_dataset_test,
                batch_size=50,
                num_workers=30,
                pin_memory=True
            )

            if not args.best_vq_vae_ckpt:
                model = VQVAE(
                in_channels=1,
                out_channels=1,
                emb_channels=512,
                num_embeddings=8192,
                spatial_dims=2,
                hid_chs=[64, 128, 256, 512],
                kernel_sizes=[3,  3, 3, 3],
                strides=[1, 2, 2, 2],
                embedding_loss_weight=1,
                beta=1,
                loss=torch.nn.L1Loss,
                deep_supervision=1,
                use_attention='none',
                sample_every_n_steps=50,
                )
            else:
                model = VQGAN(
                in_channels=1,
                out_channels=1,
                emb_channels=512,
                num_embeddings=8192,
                spatial_dims=2,
                hid_chs=[64, 128, 256, 512],
                kernel_sizes=[3,  3, 3, 3],
                strides=[1, 2, 2, 2],
                embedding_loss_weight=1,
                beta=1,
                pixel_loss=torch.nn.L1Loss,
                deep_supervision=1,
                use_attention='none',
                sample_every_n_steps=50,
                )

                model.vqvae.load_pretrained(args.best_vq_vae_ckpt)
    
    elif args.mode == "3D":
        lidc_dataset_train = LIDCDataset(
        root_dir=args.data_dir, augmentation=False, split='train')

        lidc_dataset_val = LIDCDataset(
        root_dir=args.data_dir, augmentation=False, split='val')

        lidc_dataset_test = LIDCDataset(
        root_dir=args.data_dir, augmentation=False, split='test')

        dm = SimpleDataModule(
        ds_train=lidc_dataset_train,
        ds_val=lidc_dataset_val,
        ds_test=lidc_dataset_test,
        batch_size=2,
        num_workers=30,
        pin_memory=True
        )

        if not args.best_vq_vae_ckpt:
            model = VQVAE(
            in_channels=1,
            out_channels=1,
            emb_channels=256,
            num_embeddings=8192,
            spatial_dims=3,
            hid_chs=[32, 64,  128, 256],
            kernel_sizes=[3,  3,   3, 3],
            strides=[1,  2,   2, 2],
            embedding_loss_weight=1,
            beta=1,
            loss=torch.nn.L1Loss,
            deep_supervision=0,
            use_attention='none',
            norm_name=("GROUP", {'num_groups': 4, "affine": True}),
            sample_every_n_steps=200,
            )
        else:
            model = VQGAN(
            in_channels=1,
            out_channels=1,
            emb_channels=256,
            num_embeddings=8192,
            spatial_dims=3,
            hid_chs=[32, 64,  128, 256],
            kernel_sizes=[3,  3,   3, 3],
            strides=[1,  2,   2, 2],
            embedding_loss_weight=1,
            beta=1,
            pixel_loss=torch.nn.L1Loss,
            deep_supervision=0,
            use_attention='none',
            norm_name=("GROUP", {'num_groups': 4, "affine": True}),
            sample_every_n_steps=200,
            )

            model.vqvae.load_pretrained(args.best_vq_vae_ckpt)

    ##############################################

    # -------------- Training Initialization ---------------
    to_monitor = "val/ssim_epoch"  # "train/L1"  # "val/loss"
    min_max = "max"
    save_and_sample_every = 50

    early_stopping = EarlyStopping(
        monitor=to_monitor,
        min_delta=0.0,  # minimum change in the monitored quantity to qualify as an improvement
        patience=30,  # number of checks with no improvement
        mode=min_max
    )
    checkpointing = ModelCheckpoint(
        dirpath=str(path_run_dir),  # dirpath
        monitor=to_monitor,
        every_n_train_steps=save_and_sample_every,
        save_last=True,
        save_top_k=1,
        mode=min_max,
    )
    trainer = Trainer(
        accelerator='gpu',
        devices=[1],
        # precision=16,
        # amp_backend='apex',
        # amp_level='O2',
        # gradient_clip_val=0.5,
        default_root_dir=str(path_run_dir),
        callbacks=[checkpointing],
        # callbacks=[checkpointing, early_stopping],
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
        log_every_n_steps=save_and_sample_every,
        auto_lr_find=False,
        # limit_train_batches=1000,
        # limit_val_batches=0,  # 0 = disable validation - Note: Early Stopping no longer available
        min_epochs=100,
        max_epochs=1001,
        num_sanity_val_steps=2,
    )

    # ---------------- Execute Training ----------------
    trainer.fit(model, datamodule=dm)

    # ------------- Save path to best model -------------
    model.save_best_checkpoint(
        trainer.logger.log_dir, checkpointing.best_model_path)
