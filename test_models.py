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

import torch
import torch.nn as nn

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

names_of_experiments = ["mstpp_v1","mstpp_v1_with_flash", "hscnn_v1", "hscnn_v1_with_flash"]
models = [MSTpp(in_channels=3, out_channels=31),
          MSTpp(in_channels=6, out_channels=31),
          HSCNNp(in_channels=3, out_channels=31),
          HSCNNp(in_channels=6, out_channels=31)]

for i, name in enumerate(names_of_experiments):
    criterion = HSILoss(alpha=0.5).to(device)
    weights_path = f"checkpoints/{name}/best_model.pth"
    model = models[i].to(device)
    model.load_state_dict(torch.load(weights_path))
    if i % 2 == 0:
        val_metrics = val_step(model, test_loader, criterion, device, flash=False)
    else:
        val_metrics = val_step(model, test_loader, criterion, device, flash=True)

    print(name)
    print(f"Test Loss: {val_metrics['Loss']:.4f}")
    print(f"Test SAM:  {val_metrics['SAM_deg']:.4f}")
    print(f"Test NSE:  {val_metrics['NSE']:.4f}")