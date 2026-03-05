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
from torch import newaxis

from tqdm.auto import tqdm

from skimage.color import xyz2rgb

def apply_attenuation_light(img, x0, y0, kc=1.0, kl=0.01, kq=0.01):
    h, w, c = img.shape
    y, x = np.ogrid[:h, :w]
    d = np.sqrt((x - x0)**2 + (y - y0)**2)

    attenuation = 3.0 / (kc + kl * d + kq * (d**2))

    return img * attenuation[:, :, np.newaxis]


