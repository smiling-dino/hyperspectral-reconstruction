from unittest.mock import patch

import numpy as np
import pandas as pd

import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

import json

from astropy.extern.configobj.validate import is_list
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from scipy.optimize import minimize

from tqdm.auto import tqdm

from skimage.color import xyz2rgb

import torch
from torch.utils.data import Dataset, DataLoader
import random
from tqdm import tqdm

import torch.nn as nn
from torch import Tensor

from models import HSCNNp
from models import MSTpp

munsell = pd.read_csv("data/munsell.csv")
munsell.set_index('wavelength', inplace=True)
# Нет смысла в интерполяции тк мы берём точки, которые есть в датасете
munsell = munsell.loc[range(400, 710, 10)]

illium = pd.read_csv("data/illum.csv")

illium.set_index('wavelength', inplace=True)
illium = illium.loc[400:700:10]

illium = illium / illium.sum(axis=0)

sensor = pd.read_csv('data/xyz_matching_fun.csv')
sensor.set_index('wavelength', inplace=True)
sensor = sensor.loc[range(400, 710, 10)]

k = 0.25

def make_one_patch(light, power_of_light ,reflect, sensor):
    response = 10 * power_of_light * k * (sensor.T @ (reflect.values * light)).T
    patch = np.broadcast_to(response, (32, 32, 3))
    return patch

def make_palette(responses):
    """
    responses: (16, 3)
    """
    patches = np.tile(responses[:, np.newaxis, np.newaxis, :], (1, 32, 32, 1))
    # (16, 32, 32, 3)
    palette = patches.reshape(4, 4, 32, 32, 3)
    # (4, 4, 32, 32, 3) -> (4*32, 4*32, 3)
    palette = palette.transpose(0, 2, 1, 3, 4).reshape(128, 128, 3)

    return palette

def calc_responses(light, power_of_light ,reflect, sensor):
    responses = 10 * power_of_light * k * (sensor.T @ (reflect.values * light.values[:, np.newaxis])).T
    return responses

def apply_attenuation_light(img, x0, y0, kc=1.0, kl=0.005, kq=0.05):
    h, w, c = img.shape
    y, x = np.ogrid[:h, :w]
    d = np.sqrt((x - x0)**2 + (y - y0)**2)

    attenuation = 1.0 / (kc + kl * d + kq * (d**2))

    return img * attenuation[:, :, np.newaxis]

def make_palette_with_flashlight(light ,reflect, sensor):
    """
    мне влом(
    """
    flashlight_mask = make_palette(calc_responses(illium['L-B1'], 100, reflect, sensor).to_numpy())
    flashlight_palette = apply_attenuation_light(flashlight_mask, 64,64)
    palette = flashlight_palette + make_palette(calc_responses(light, 1, reflect, sensor).to_numpy())

    return palette

train_val_reflect, test_reflect = train_test_split(munsell.T, test_size=112, random_state=42)
train_reflect, val_reflect = train_test_split(train_val_reflect, test_size=112, random_state=42)

class SpectrumDataset(Dataset):
    def __init__(self, reflect_df):
        self.reflect = torch.tensor(reflect_df.values, dtype=torch.float32)

    def __len__(self):
        return len(self.reflect)

    def __getitem__(self, idx):
        return self.reflect[idx]


train_dataset = SpectrumDataset(train_reflect)
val_dataset = SpectrumDataset(val_reflect)
test_dataset = SpectrumDataset(test_reflect)

# 512 спектров за одну итерацию
palettes_in_batch = 16
reflect_per_batch = palettes_in_batch * 8

train_loader = DataLoader(
    train_dataset,
    batch_size=reflect_per_batch,
    shuffle=True,
    drop_last=True,
    num_workers=4
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    drop_last=False,
    num_workers=4
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    drop_last=False,
    num_workers=4
)


class PaletteRenderer:
    def __init__(self, sensor_df, ambient_dict, flash_series, device='cuda:0'):
        self.device = device

        # [31, 3]
        self.sensor = torch.tensor(sensor_df.values, dtype=torch.float32, device=device)

        # [31]
        self.flashlight = torch.tensor(flash_series.values, dtype=torch.float32, device=device)

        ambient_list = [ambient_dict[name].values for name in ambient_dict.keys()]
        self.ambients = torch.tensor(ambient_list, dtype=torch.float32, device=device)
        self.num_lights = self.ambients.shape[0]

        # Сетка для фонарика можем по ней определять расстояния от центра
        y, x = torch.meshgrid(torch.linspace(0, 127, 128, device=device),
                              torch.linspace(0, 127, 128, device=device), indexing='ij')
        self.y_grid = y.view(1, 128, 128, 1) # [1, 128, 128, 1]
        self.x_grid = x.view(1, 128, 128, 1) # [1, 128, 128, 1]

    def _to_palette(self, features):
        """
        [B, 16, C] -> [B, 128, 128, C]
        """
        B, num_patches, C = features.shape
        grid = features.view(B, 4, 4, C)
        # Делаем каждый патч 32*32
        grid = grid.unsqueeze(3).unsqueeze(5).expand(-1, 4, 4, 32, C, 32)
        return grid.permute(0, 1, 3, 2, 5, 4).reshape(B, 128, 128, C)
# TODO поменять k
    def render_batch(self, spectra_batch, k_exp=0.25, flash_power=100.0):
        """
        spectra_batch: тензор [B, 16, 31] (исходные спектры Манселла)
        Возвращает:
            X: [B, 6, 128, 128] (Вход модели: 3 канала со вспышкой + 3 амбиент)
            y: [B, 31, 128, 128] (Выход модели: истинный спектральный куб)
        """
        B = spectra_batch.shape[0]

        # XYZ
        light_idx = torch.randint(0, self.num_lights, (B,), device=self.device)
        batch_ambients = self.ambients[light_idx].unsqueeze(1) # [B, 1, 31]

        xyz_amb = 10 * k_exp * ((spectra_batch * batch_ambients) @ self.sensor)
        xyz_flash = 10 * k_exp * flash_power * ((spectra_batch * self.flashlight.view(1, 1, 31)) @ self.sensor)

        pal_amb = self._to_palette(xyz_amb)     # [B, 128, 128, 3]
        pal_flash = self._to_palette(xyz_flash) # [B, 128, 128, 3]

        # пока фонарик в центре, но потом можем поменять
        x0 = 64.0
        y0 = 64.0

        dist_sq = (self.x_grid - x0)**2 + (self.y_grid - y0)**2
        attenuation = 1.0 / (1.0 + 0.005 * torch.sqrt(dist_sq) + 0.05 * dist_sq)

        # merge with flashlight
        X_flash_rgb = torch.clamp(pal_amb + (pal_flash * attenuation), 0.0, 1.0)
        X_amb_rgb = torch.clamp(pal_amb, 0.0, 1.0)

        # Формат PyTorch [B, C, H, W]
        X_flash_img = X_flash_rgb.permute(0, 3, 1, 2)
        X_amb_img = X_amb_rgb.permute(0, 3, 1, 2)

        # Склеиваем по оси каналов (dim=1). Получаем 6 каналов.
        X_6chan = torch.cat([X_flash_img, X_amb_img], dim=1)

        # Спектральная картинка - y
        # Просто растягиваем исходные 31-канальные спектры в размер 128x128
        y_spectra_img = self._to_palette(spectra_batch)
        y_hsi = y_spectra_img.permute(0, 3, 1, 2)

        return X_6chan, y_hsi

    def render_no_flash_batch(self, spectra_batch, k_exp=0.25):
        """
        spectra_batch: тензор [B, 16, 31] (исходные спектры Манселла)
        Возвращает:
            X: [B, 6, 128, 128] (Вход модели: 3 амбиент)
            y: [B, 31, 128, 128] (Выход модели: истинный спектральный куб)
        """
        B = spectra_batch.shape[0]

        # XYZ
        light_idx = torch.randint(0, self.num_lights, (B,), device=self.device)
        batch_ambients = self.ambients[light_idx].unsqueeze(1) # [B, 1, 31]

        xyz_amb = 10 * k_exp * ((spectra_batch * batch_ambients) @ self.sensor)

        pal_amb = self._to_palette(xyz_amb)     # [B, 128, 128, 3]

        X_amb_img = torch.clamp(pal_amb, 0.0, 1.0)

        # Формат PyTorch [B, C, H, W]
        X_amb_img = X_amb_img.permute(0, 3, 1, 2)

        # Спектральная картинка - y
        # Просто растягиваем исходные 31-канальные спектры в размер 128x128
        y_spectra_img = self._to_palette(spectra_batch)
        y_hsi = y_spectra_img.permute(0, 3, 1, 2)

        return X_amb_img, y_hsi

def pixelwise_spectral_angle_mapper(preds: Tensor, target: Tensor, eps=1e-8):
    products = (preds * target).sum(dim=-3)
    magnitudes = preds.norm(dim=-3) * target.norm(dim=-3) + eps
    cosine_sim = torch.clamp(products / magnitudes, -1.0 + eps, 1.0 - eps)

    return torch.rad2deg(torch.acos(cosine_sim))

def pixelwise_normalized_spectral_error(preds: Tensor, target: Tensor, eps=1e-8):
    diff = (preds - target).abs().sum(dim=-3)
    scale = target.abs().sum(dim=-3) + eps
    return diff / scale

class HSILoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, preds, target):
        nse_loss = pixelwise_normalized_spectral_error(preds, target).mean()
        sam_loss = pixelwise_spectral_angle_mapper(preds, target).mean()

        return self.alpha * nse_loss + (1.0 - self.alpha) * sam_loss

device = torch.device('cuda:1')
# metric_deltaE = DeltaE().to(device)

sensor_gpu = torch.tensor(sensor.values, dtype=torch.float32, device=device)
illuminant_d65 = torch.tensor(illium['D65'].values, dtype=torch.float32, device=device)

selected_ambients = {
    'D65': illium['D65'],
    'D50': illium['D50'],
    'A18': illium['A18'],
    'F2': illium['F2'],
    'F11': illium['F11'],
    'L-V1': illium['L-V1'],
    'L-V2': illium['L-V2']
   # 'Flashlight': illium['L-B1']
}

renderer = PaletteRenderer(
    sensor_df=sensor,
    ambient_dict=selected_ambients,
    flash_series=illium['L-B1'],
    device='cuda:1'
)

def train_step(model, dataloader, renderer, optimizer, criterion, device, flash=True):
    model.train()
    running_loss = 0.0
    pbar = tqdm(dataloader, desc="Training")

    for spectra in pbar:
        spectra = spectra.to(device)

        spectra_palettes = spectra.view(-1, 16, 31)

        if flash:
            X_batch, y_target = renderer.render_batch(spectra_palettes)
        else:
            X_batch, y_target = renderer.render_no_flash_batch(spectra_palettes)

        optimizer.zero_grad()

        preds = model(X_batch)
        loss = criterion(preds, y_target)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

    return running_loss / len(dataloader)

def val_step(model, dataloader, criterion, device, flash=True):
    model.eval()

    val_loss = 0.0
    total_sam = 0.0
    total_nse = 0.0

   # metric_deltaE.reset()

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")

        for spectra in pbar:
            spectra = spectra.to(device)

            spectra_palettes = spectra.view(-1, 16, 31)

            if flash:
                X_batch, y_target = renderer.render_batch(spectra_palettes)
            else:
                X_batch, y_target = renderer.render_no_flash_batch(spectra_palettes)

            preds = model(X_batch)

            loss = criterion(preds, y_target)
            val_loss += loss.item()

            batch_sam = pixelwise_spectral_angle_mapper(preds, y_target).mean().item()
            batch_nse = pixelwise_normalized_spectral_error(preds, y_target).mean().item()

            total_sam += batch_sam
            total_nse += batch_nse

          #  preds_rgb = xyz2rgb()
          #  target_rgb = xyz2rgb()

            # metric_deltaE.update(preds_rgb, target_rgb)

    num_batches = len(dataloader)
    metrics = {
        'Loss': val_loss / num_batches,
        'SAM_deg': total_sam / num_batches,
        'NSE': total_nse / num_batches,
       # 'DeltaE': metric_deltaE.compute().item()
    }

    return metrics

names_of_experiments = ["mstpp_v1",
                        "mstpp_v1_with_flash",
                        "hscnn_v1",
                        "hscnn_v1_with_flash"]

models = [MSTpp(in_channels=3, out_channels=31),
          MSTpp(in_channels=6, out_channels=31),
          HSCNNp(in_channels=3, out_channels=31),
          HSCNNp(in_channels=6, out_channels=31)]

models_dict = {}
for name in names_of_experiments:
    is_flash = "flash" in name
    in_channels = 6 if is_flash else 3

    if "mstpp" in name:
        model = MSTpp(in_channels=in_channels, out_channels=31).to(device)
    else:
        model = HSCNNp(in_channels=in_channels, out_channels=31).to(device)

    weights_path = f"./checkpoints/{name}/best_model.pth"
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    models_dict[name] = {"model": model, "is_flash": is_flash}

num_models = len(names_of_experiments)
sns.set_theme('paper')
def show_error_maps():
    for batch_idx, spectra in enumerate(test_loader):
        spectra = spectra.to(device)
        spectra_palettes = spectra.view(1, 16, 31)

        fig, axes = plt.subplots(num_models, 3, figsize=(8, 2.5 * num_models))
        fig.suptitle(f"Error Maps for Test Patch #{batch_idx + 1}", fontsize=16)

        for idx, (name, config) in enumerate(models_dict.items()):
            model = config["model"]
            is_flash = config["is_flash"]

            torch.manual_seed(42 + batch_idx)

            if is_flash:
                X_batch, y_target = renderer.render_batch(spectra_palettes)
                input_rgb = X_batch[0, :3].permute(1, 2, 0).cpu().numpy()
            else:
                X_batch, y_target = renderer.render_no_flash_batch(spectra_palettes)
                input_rgb = X_batch[0, :3].permute(1, 2, 0).cpu().numpy()

            with torch.no_grad():
                preds = model(X_batch)

            nse_map = pixelwise_normalized_spectral_error(preds, y_target)[0].cpu().numpy()
            sam_map = pixelwise_spectral_angle_mapper(preds, y_target)[0].cpu().numpy()

            ax_rgb = axes[idx, 0]
            ax_nse = axes[idx, 1]
            ax_sam = axes[idx, 2]

            ax_rgb.imshow(input_rgb)
            ax_rgb.set_title(f"RGB ({name})")
            ax_rgb.axis('off')

            im_nse = ax_nse.imshow(nse_map, cmap='magma', vmin=0.0, vmax=1.0)
            ax_nse.set_title(f"NSE ({name})")
            ax_nse.axis('off')
            fig.colorbar(im_nse, ax=ax_nse, fraction=0.046, pad=0.04)

            im_sam = ax_sam.imshow(sam_map, cmap='magma', vmin=1.5, vmax=8.0)
            ax_sam.set_title(f"SAM [deg] ({name})")
            ax_sam.axis('off')
            fig.colorbar(im_sam, ax=ax_sam, fraction=0.046, pad=0.04)

        # plt.tight_layout(h_pad=0.5, w_pad=0.5, rect=[0, 0.03, 1, 0.95])
        plt.show()
        # break


def show_error_maps_v2():
    for batch_idx, spectra in enumerate(test_loader):
        spectra = spectra.to(device)
        spectra_palettes = spectra.view(1, 16, 31)

        fig, axes = plt.subplots(2, 5, figsize=(18, 7))
        fig.suptitle(f"Error Maps for Test Patch #{batch_idx + 1}", fontsize=16)

        for row_idx, is_flash in enumerate([False, True]):
            torch.manual_seed(42 + batch_idx)

            if is_flash:
                X_batch, y_target = renderer.render_batch(spectra_palettes)
                row_label = "Flash"
            else:
                X_batch, y_target = renderer.render_no_flash_batch(spectra_palettes)
                row_label = "No Flash"

            input_rgb = X_batch[0, :3].permute(1, 2, 0).cpu().numpy()

            ax_rgb = axes[row_idx, 0]
            ax_rgb.imshow(input_rgb)
            ax_rgb.set_title(f"Input RGB ({row_label})")
            ax_rgb.axis('off')

            hscnn_name, hscnn_model = None, None
            mstpp_name, mstpp_model = None, None

            for name, config in models_dict.items():
                if config["is_flash"] == is_flash:
                    if "mstpp" in name:
                        mstpp_name, mstpp_model = name, config["model"]
                    else:
                        hscnn_name, hscnn_model = name, config["model"]

            if hscnn_model is not None:
                with torch.no_grad():
                    preds_hscnn = hscnn_model(X_batch)

                nse_hscnn = pixelwise_normalized_spectral_error(preds_hscnn, y_target)[0].cpu().numpy()
                sam_hscnn = pixelwise_spectral_angle_mapper(preds_hscnn, y_target)[0].cpu().numpy()

                ax_nse_h = axes[row_idx, 1]
                im_nse_h = ax_nse_h.imshow(nse_hscnn, cmap='magma', vmin=0.0, vmax=1.0)
                ax_nse_h.set_title(f"NSE ({hscnn_name})")
                ax_nse_h.axis('off')
                fig.colorbar(im_nse_h, ax=ax_nse_h, fraction=0.046, pad=0.04, shrink=0.8)

                ax_sam_h = axes[row_idx, 2]
                im_sam_h = ax_sam_h.imshow(sam_hscnn, cmap='magma', vmin=1.5, vmax=8.0)
                ax_sam_h.set_title(f"SAM [deg] ({hscnn_name})")
                ax_sam_h.axis('off')
                fig.colorbar(im_sam_h, ax=ax_sam_h, fraction=0.046, pad=0.04, shrink=0.8)

            if mstpp_model is not None:
                with torch.no_grad():
                    preds_mstpp = mstpp_model(X_batch)

                nse_mstpp = pixelwise_normalized_spectral_error(preds_mstpp, y_target)[0].cpu().numpy()
                sam_mstpp = pixelwise_spectral_angle_mapper(preds_mstpp, y_target)[0].cpu().numpy()

                ax_nse_m = axes[row_idx, 3]
                im_nse_m = ax_nse_m.imshow(nse_mstpp, cmap='magma', vmin=0.0, vmax=1.0)
                ax_nse_m.set_title(f"NSE ({mstpp_name})")
                ax_nse_m.axis('off')
                fig.colorbar(im_nse_m, ax=ax_nse_m, fraction=0.046, pad=0.04, shrink=0.8)

                ax_sam_m = axes[row_idx, 4]
                im_sam_m = ax_sam_m.imshow(sam_mstpp, cmap='magma', vmin=1.5, vmax=8.0)
                ax_sam_m.set_title(f"SAM [deg] ({mstpp_name})")
                ax_sam_m.axis('off')
                fig.colorbar(im_sam_m, ax=ax_sam_m, fraction=0.046, pad=0.04, shrink=0.8)

        plt.tight_layout(h_pad=1.0, w_pad=0.5, rect=[0, 0.03, 1, 0.95])
        plt.show()
        # break


import torch
import matplotlib.pyplot as plt

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


def show_flash_failure_maps():
    for batch_idx, spectra in enumerate(test_loader):
        spectra = spectra.to(device)
        spectra_palettes = spectra.view(1, 16, 31)

        fig, axes = plt.subplots(2, 5, figsize=(18, 7))
        fig.suptitle(f"Error Maps (Simulated FLASH FAILURE) - Test Patch #{batch_idx + 1}", fontsize=16)

        for row_idx, is_flash in enumerate([False, True]):
            torch.manual_seed(42 + batch_idx)

            if is_flash:
                X_batch, y_target = renderer.render_batch(spectra_palettes)

                X_batch[:, 3:, :, :] = 0.0
                row_label = "Flash ZEROED!"
            else:
                X_batch, y_target = renderer.render_no_flash_batch(spectra_palettes)
                row_label = "No Flash Model"

            input_rgb = X_batch[0, :3].permute(1, 2, 0).cpu().numpy()

            ax_rgb = axes[row_idx, 0]
            ax_rgb.imshow(input_rgb)
            ax_rgb.set_title(f"Input RGB\n({row_label})", fontsize=11)
            ax_rgb.axis('off')

            hscnn_name, hscnn_model = None, None
            mstpp_name, mstpp_model = None, None

            for name, config in models_dict.items():
                if config["is_flash"] == is_flash:
                    if "mstpp" in name:
                        mstpp_name, mstpp_model = name, config["model"]
                    else:
                        hscnn_name, hscnn_model = name, config["model"]

            if hscnn_model is not None:
                with torch.no_grad():
                    preds_hscnn = hscnn_model(X_batch)

                nse_hscnn = pixelwise_normalized_spectral_error(preds_hscnn, y_target)[0].cpu().numpy()
                sam_hscnn = pixelwise_spectral_angle_mapper(preds_hscnn, y_target)[0].cpu().numpy()

                ax_nse_h = axes[row_idx, 1]
                im_nse_h = ax_nse_h.imshow(nse_hscnn, cmap='magma', vmin=0.0, vmax=1.0)
                ax_nse_h.set_title(f"NSE ({hscnn_name})", fontsize=10)
                ax_nse_h.axis('off')
                fig.colorbar(im_nse_h, ax=ax_nse_h, fraction=0.046, pad=0.04, shrink=0.8)

                ax_sam_h = axes[row_idx, 2]
                im_sam_h = ax_sam_h.imshow(sam_hscnn, cmap='magma', vmin=1.5, vmax=8.0)
                ax_sam_h.set_title(f"SAM [deg] ({hscnn_name})", fontsize=10)
                ax_sam_h.axis('off')
                fig.colorbar(im_sam_h, ax=ax_sam_h, fraction=0.046, pad=0.04, shrink=0.8)

            if mstpp_model is not None:
                with torch.no_grad():
                    preds_mstpp = mstpp_model(X_batch)

                nse_mstpp = pixelwise_normalized_spectral_error(preds_mstpp, y_target)[0].cpu().numpy()
                sam_mstpp = pixelwise_spectral_angle_mapper(preds_mstpp, y_target)[0].cpu().numpy()

                ax_nse_m = axes[row_idx, 3]
                im_nse_m = ax_nse_m.imshow(nse_mstpp, cmap='magma', vmin=0.0, vmax=1.0)
                ax_nse_m.set_title(f"NSE ({mstpp_name})", fontsize=10)
                ax_nse_m.axis('off')
                fig.colorbar(im_nse_m, ax=ax_nse_m, fraction=0.046, pad=0.04, shrink=0.8)

                ax_sam_m = axes[row_idx, 4]
                im_sam_m = ax_sam_m.imshow(sam_mstpp, cmap='magma', vmin=1.5, vmax=8.0)
                ax_sam_m.set_title(f"SAM [deg] ({mstpp_name})", fontsize=10)
                ax_sam_m.axis('off')
                fig.colorbar(im_sam_m, ax=ax_sam_m, fraction=0.046, pad=0.04, shrink=0.8)

        plt.tight_layout(h_pad=1.0, w_pad=0.5, rect=[0, 0.03, 1, 0.95])
        plt.show()

        # break  # Оставляем один патч для быстрой проверки


# show_flash_failure_maps()
# show_error_maps()

show_error_maps_v2()

def show_spectra_grid_4x4():
    wavelengths = np.arange(400, 710, 10)

    for batch_idx, spectra in enumerate(test_loader):
        spectra = spectra.to(device)
        spectra_palettes = spectra.view(1, 16, 31)

        patch_size = 32
        center_pixels = []
        for row in range(4):
            for col in range(4):
                cy = row * patch_size + patch_size // 2
                cx = col * patch_size + patch_size // 2
                center_pixels.append((cy, cx))

        predictions = {coord: {} for coord in center_pixels}
        ground_truths = {}

        for name, config in models_dict.items():
            model = config["model"]
            is_flash = config["is_flash"]

            torch.manual_seed(42 + batch_idx)

            if is_flash:
                X_batch, y_target = renderer.render_batch(spectra_palettes)
            else:
                X_batch, y_target = renderer.render_no_flash_batch(spectra_palettes)

            with torch.no_grad():
                preds = model(X_batch)

            for (y, x) in center_pixels:
                predictions[(y, x)][name] = preds[0, :, y, x].cpu().numpy()
                if (y, x) not in ground_truths:
                    ground_truths[(y, x)] = y_target[0, :, y, x].cpu().numpy()

        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        fig.suptitle(f"Спектры для 16 патчей (Палитра #{batch_idx + 1})", fontsize=20, y=1.02)

        colors = {
            "mstpp_v1": "blue",
            "mstpp_v1_with_flash": "cyan",
            "hscnn_v1": "red",
            "hscnn_v1_with_flash": "orange"
        }

        for idx, ((y, x), ax) in enumerate(zip(center_pixels, axes.flatten())):
            ax.plot(wavelengths, ground_truths[(y, x)], color='black', linewidth=3, linestyle='--',
                    label='Ground Truth')

            for name in names_of_experiments:
                ax.plot(wavelengths, predictions[(y, x)][name], color=colors[name], linewidth=2, alpha=0.8, label=name)

            ax.set_title(f"Патч {idx + 1} (y={y}, x={x})", fontsize=12)
            ax.set_xlabel("Длина волны (нм)", fontsize=10)
            ax.set_ylabel("Отражение", fontsize=10)
            ax.grid(True, linestyle=':', alpha=0.7)

            if idx == 0:
                ax.legend(loc='best', fontsize=10)

        plt.tight_layout()
        plt.show()

        #break


# show_spectra_grid_4x4()