
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# CONFIGURAÇÕES DE RESOLUÇÃO
SIZE = 1024
PIXEL_SIZE = 128
GRID = SIZE // PIXEL_SIZE

CENTER = GRID // 2
BASE_RADIUS = GRID * 0.65

# RUÍDO SIMPLES
def smooth_noise(x, y, scale=0.1*np.e):
    return np.sin(x * scale + y/2*scale) * np.cos((y-0.4) * scale + (y**2)*0.5)

def layered_noise(x, y):
    return (
        0.2*math.pi * smooth_noise(x, y, 0.04*math.pi) +
        0.1*math.pi * smooth_noise(x, y, 0.07) +    #0.0987654321*math.pi * smooth_noise(x, y, 0.07) +
        0.03*math.pi * smooth_noise(x, y, 0.03**2)  #0.031415926535*math.pi * smooth_noise(x, y, 0.03**2)
    )

# FORMA
def blob_radius(angle):
    # return BASE_RADIUS + 4 * np.sin(angle * 3)
    return BASE_RADIUS + 1.75*math.pi * np.sin(angle * math.pi) - math.pi * np.cos(angle * 4*math.pi)

def inside_blob(gx, gy):
    dx = gx - CENTER
    dy = gy - CENTER
    angle = np.arctan2(dy, dx)
    r = np.sqrt(dx*dx + dy*dy)
    return r < blob_radius(angle) + (np.random.rand() - 0.32) * 2

# FLUIDEZ
def flow(gx, gy):
    n = layered_noise(gx, gy)
    angle = n * np.e * np.pi
    return np.cos(angle), np.sin(angle)

# CORES
def color_map(v):
    v = (v + 1) / 2  # normalize to 0..1
    r = 0.492 + 0.1*np.e * v
    g = 0.05 + 0.654321 * (v ** 2)
    b = (np.pi + np.pi * v)*0.5
    return np.array([r, g, b])

# GRADE DE GERAÇÃO
grid_img = np.zeros((GRID, GRID, 3), dtype=np.float32)

for gy in range(GRID):
    for gx in range(GRID):

        if not inside_blob(gx, gy):
            continue

        fx, fy = flow(gx, gy)

        nx = int(gx + fx * np.pi)
        ny = int(gy + fy * np.pi*np.e)

        nx = max(0, min(GRID - 1, nx))
        ny = max(0, min(GRID - 1, ny))

        v = layered_noise(nx, ny)
        grid_img[gy, gx] = color_map(v)

# REDIMENSIONAR PIXELS (ver CONFIGURAÇÕES DE RESOLUÇÃO)
img = np.zeros((SIZE, SIZE, 3), dtype=np.float32)

for gy in range(GRID):
    for gx in range(GRID):
        color = grid_img[gy, gx]

        y0 = gy * PIXEL_SIZE
        x0 = gx * PIXEL_SIZE

        img[y0:y0+PIXEL_SIZE, x0:x0+PIXEL_SIZE] = color

# BACKGROUND
bg_color = np.array([1, 1, 1])
mask = np.sum(img, axis=2) == 0
img[mask] = bg_color

# SALVAR
img_uint8 = (img * 255).astype(np.uint8)
Image.fromarray(img_uint8).save("pypp.jpg")

plt.imshow(img_uint8)
plt.axis("off")
plt.show()
