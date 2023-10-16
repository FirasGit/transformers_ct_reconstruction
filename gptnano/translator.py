import torch
import torch.nn as nn
import torch.nn.functional as F
from nanogpt import GPT
from vqgan.model import VQGAN
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass
from collections import OrderedDict


def rename_state_dict_keys(state_dict, key_transformation):
    """
    Rename the keys of a state dict.
    state_dict         -> The saved state dict.
    key_transformation -> Function that accepts the old key names of the state
                          dict as the only argument and returns the new key name.
    """
    new_state_dict = OrderedDict()

    for key, value in state_dict.items():
        new_key = key_transformation(key)
        new_state_dict[new_key] = value
    
    return new_state_dict


class VQGANTransformer(nn.Module):
    def __init__(self, args):
        super(VQGANTransformer, self).__init__()

        self.sos_token = args.sos_token

        self.vqgan_3d = self.load_vqgan_3d(args)
        self.vqgan_2d = self.load_vqgan_2d(args)

        @dataclass
        class TransformerConfig:
            block_size: int = args.block_size
            vocab_size: int = args.num_codebook_vectors
            n_layer: int = 8
            n_head: int = 8
            n_embd: int = 512
            dropout: float = 0.1
            bias: bool = False

        self.transformer = GPT(TransformerConfig)

        self.pkeep = args.pkeep

    def load_gpt(self, args, strict=True):
        checkpoint_path = Path(args.checkpoint_path_gpt)
        checkpoint = torch.load(checkpoint_path)
        checkpoint = rename_state_dict_keys(checkpoint, lambda x: '.'.join(x.split('.')[1:]) if x.startswith('_orig_mod') else x)
        self.load_state_dict(checkpoint, strict=strict)
        self.transformer = self.transformer.eval()

    @staticmethod
    def load_vqgan_3d(args):
        # TODO: This shouldn't be hardcoded
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
        checkpoint_path = Path(args.checkpoint_path_3d_vqgan)
        model.load_pretrained(checkpoint_path)
        model = model.eval()
        return model

    @staticmethod
    def load_vqgan_2d(args):
        # TODO: This shouldn't be hardcoded
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
        checkpoint_path = Path(args.checkpoint_path_2d_vqgan)
        model.load_pretrained(checkpoint_path)
        model = model.eval()
        return model

    @torch.no_grad()
    def encode_to_z(self, x, use_3d=True):
        if use_3d:
            indices, _ = self.vqgan_3d.vqvae.encode_to_indices(x)
        else:
            indices, _ = self.vqgan_2d.vqvae.encode_to_indices(x)
        return indices

    @torch.no_grad()
    def z_to_image(self, indices, ch=8, p1=16, p2=16, p3=16, use_3d=True):
        if use_3d:
            image = self.vqgan_3d.vqvae.decode_from_indices(
                indices, (1, p1, p2, p3, ch))
        else:
            image = self.vqgan_2d.vqvae.decode_from_indices(
                indices, (1, p1, p2, ch))
        return image

    def forward(self, x, use_indices=True):
        if use_indices:
            indices_ct, indices_ap, indices_lat = x
        else:
            # TODO: Encode all images to z (3d and 2d)
            #indices = self.encode_to_z(x)
            raise NotImplementedError("Encoding to z is not implemented yet")

        sos_tokens = torch.ones(indices_ct.shape[0], 1) * self.sos_token
        sos_tokens = sos_tokens.long().to("cuda")

        mask = torch.bernoulli(
            self.pkeep * torch.ones(indices_ct.shape, device=indices_ct.device))
        mask = mask.round().to(dtype=torch.int64)
        random_indices = torch.randint_like(
            indices_ct, self.transformer.config.vocab_size)
        new_indices = mask * indices_ct + (1 - mask) * random_indices

        new_indices = torch.cat(
            (sos_tokens, indices_ap, indices_lat, new_indices), dim=1)

        target = torch.cat((indices_ap, indices_lat, indices_ct), dim=1)

        logits, _ = self.transformer(new_indices[:, :-1])

        return logits, target

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float("inf")
        return out

    @torch.no_grad()
    def sample(self, x, c, steps, temperature=1.0, top_k=100):
        self.transformer.eval()
        x = torch.cat((c, x), dim=1)
        for k in tqdm(range(steps)):
            logits, _ = self.transformer(x)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)

            probs = F.softmax(logits, dim=-1)

            ix = torch.multinomial(probs, num_samples=1)

            x = torch.cat((x, ix), dim=1)

        x = x[:, c.shape[1]:]
        self.transformer.train()
        return x

    @torch.no_grad()
    def log_images(self, x, use_indices=True, temperature=1.0, top_k=100):
        log = dict()

        if use_indices:
            indices_ct, indices_ap, indices_lat = x
        else:
            # TODO: Encode all images to z (3d and 2d)
            #indices = self.encode_to_z(x)
            raise NotImplementedError("Encoding to z is not implemented yet")

        sos_tokens = torch.ones(indices_ct.shape[0], 1) * self.sos_token
        sos_tokens = sos_tokens.long().to("cuda")

        start_indices_ = indices_ct[:, :indices_ct.shape[1] // 2]
        start_indices = torch.cat(
            (indices_ap, indices_lat, start_indices_), dim=1)
        sample_indices = self.sample(
            start_indices, sos_tokens, steps=indices_ct.shape[1] - start_indices_.shape[1], temperature=temperature, top_k=top_k)

        sample_indices_ap = sample_indices[:, :indices_ap.shape[1]]
        sample_indices_lat = sample_indices[:, indices_ap.shape[1]
            :indices_ap.shape[1]+indices_lat.shape[1]]
        sample_indices_ct = sample_indices[:,
                                           indices_ap.shape[1]+indices_lat.shape[1]:]
        half_sample_ct = self.z_to_image(
            sample_indices_ct, use_3d=True, ch=256, p1=16, p2=16, p3=16)
        half_sample_ap = self.z_to_image(
            sample_indices_ap, use_3d=False, ch=512, p1=16, p2=16)
        half_sample_lat = self.z_to_image(
            sample_indices_lat, use_3d=False, ch=512, p1=16, p2=16)

        start_indices_ = indices_ct[:, :0]
        start_indices = torch.cat(
            (indices_ap, indices_lat, start_indices_), dim=1)
        sample_indices = self.sample(
            start_indices, sos_tokens, steps=indices_ct.shape[1], temperature=temperature, top_k=top_k)

        sample_indices_ap = sample_indices[:, :indices_ap.shape[1]]
        sample_indices_lat = sample_indices[:, indices_ap.shape[1]
            :indices_ap.shape[1]+indices_lat.shape[1]]
        sample_indices_ct = sample_indices[:,
                                           indices_ap.shape[1]+indices_lat.shape[1]:]

        full_sample_ct = self.z_to_image(
            sample_indices_ct, use_3d=True, ch=256, p1=16, p2=16, p3=16)
        full_sample_ap = self.z_to_image(
            sample_indices_ap, use_3d=False, ch=512, p1=16, p2=16)
        full_sample_lat = self.z_to_image(
            sample_indices_lat, use_3d=False, ch=512, p1=16, p2=16)

        x_rec_ct = self.z_to_image(
            indices_ct, use_3d=True, ch=256, p1=16, p2=16, p3=16)
        x_rec_ap = self.z_to_image(
            indices_ap, use_3d=False, ch=512, p1=16, p2=16)
        x_rec_lat = self.z_to_image(
            indices_lat, use_3d=False, ch=512, p1=16, p2=16)

        log["input"] = x
        log["rec_ct"] = x_rec_ct
        log["rec_ap"] = x_rec_ap
        log["rec_lat"] = x_rec_lat
        log["half_sample_ct"] = half_sample_ct
        log["half_sample_ap"] = half_sample_ap
        log["half_sample_lat"] = half_sample_lat
        log["full_sample_ct"] = full_sample_ct
        log["full_sample_ap"] = full_sample_ap
        log["full_sample_lat"] = full_sample_lat

        return log, torch.concat((x_rec_ct, half_sample_ct, full_sample_ct)), torch.concat((x_rec_ap, half_sample_ap, full_sample_ap)), torch.concat((x_rec_lat, half_sample_lat, full_sample_lat))

    @torch.no_grad()
    def log_images_monoplanar(self, x, use_indices=True, temperature=1.0, top_k=100):
        log = dict()

        if use_indices:
            indices_ct, indices_ap, indices_lat = x
        else:
            # TODO: Encode all images to z (3d and 2d)
            #indices = self.encode_to_z(x)
            raise NotImplementedError("Encoding to z is not implemented yet")

        sos_tokens = torch.ones(indices_ct.shape[0], 1) * self.sos_token
        sos_tokens = sos_tokens.long().to("cuda")

		# Monoplanar
        start_indices_ = indices_ct[:, :0]
        start_indices = torch.cat(
            (indices_ap, start_indices_), dim=1)
        sample_indices = self.sample(
            start_indices, sos_tokens, steps=indices_lat.shape[1] + indices_ct.shape[1], temperature=temperature, top_k=top_k)

        sample_indices_ap = sample_indices[:, :indices_ap.shape[1]]
        sample_indices_lat = sample_indices[:, indices_ap.shape[1]
            :indices_ap.shape[1]+indices_lat.shape[1]]
        sample_indices_ct = sample_indices[:,
                                           indices_ap.shape[1]+indices_lat.shape[1]:]
        half_sample_ct = self.z_to_image(
            sample_indices_ct, use_3d=True, ch=256, p1=16, p2=16, p3=16)
        half_sample_ap = self.z_to_image(
            sample_indices_ap, use_3d=False, ch=512, p1=16, p2=16)
        half_sample_lat = self.z_to_image(
            sample_indices_lat, use_3d=False, ch=512, p1=16, p2=16)

		# Biplanar
        start_indices_ = indices_ct[:, :0]
        start_indices = torch.cat(
            (indices_ap, indices_lat, start_indices_), dim=1)
        sample_indices = self.sample(
            start_indices, sos_tokens, steps=indices_ct.shape[1], temperature=temperature, top_k=top_k)

        sample_indices_ap = sample_indices[:, :indices_ap.shape[1]]
        sample_indices_lat = sample_indices[:, indices_ap.shape[1]
            :indices_ap.shape[1]+indices_lat.shape[1]]
        sample_indices_ct = sample_indices[:,
                                           indices_ap.shape[1]+indices_lat.shape[1]:]

        full_sample_ct = self.z_to_image(
            sample_indices_ct, use_3d=True, ch=256, p1=16, p2=16, p3=16)
        full_sample_ap = self.z_to_image(
            sample_indices_ap, use_3d=False, ch=512, p1=16, p2=16)
        full_sample_lat = self.z_to_image(
            sample_indices_lat, use_3d=False, ch=512, p1=16, p2=16)

        x_rec_ct = self.z_to_image(
            indices_ct, use_3d=True, ch=256, p1=16, p2=16, p3=16)
        x_rec_ap = self.z_to_image(
            indices_ap, use_3d=False, ch=512, p1=16, p2=16)
        x_rec_lat = self.z_to_image(
            indices_lat, use_3d=False, ch=512, p1=16, p2=16)

        log["input"] = x
        log["rec_ct"] = x_rec_ct
        log["rec_ap"] = x_rec_ap
        log["rec_lat"] = x_rec_lat
        log["half_sample_ct"] = half_sample_ct
        log["half_sample_ap"] = half_sample_ap
        log["half_sample_lat"] = half_sample_lat
        log["full_sample_ct"] = full_sample_ct
        log["full_sample_ap"] = full_sample_ap
        log["full_sample_lat"] = full_sample_lat

        return log, torch.concat((x_rec_ct, half_sample_ct, full_sample_ct)), torch.concat((x_rec_ap, half_sample_ap, full_sample_ap)), torch.concat((x_rec_lat, half_sample_lat, full_sample_lat))

