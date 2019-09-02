from math import sqrt, ceil
import numpy as np

def visualize(Xs, ubound=255.0, padding=1):
  (N, H, W, C) = Xs.shape
  value_size = int(ceil(sqrt(N)))
  value_height = H * value_size + padding * (value_size - 1)
  value_width = W * value_size + padding * (value_size - 1)
  value = np.zeros((value_height, value_width, C))
  next_idx = 0
  y0, y1 = 0, H
  for y in range(value_size):
    x0, x1 = 0, W
    for x in range(value_size):
      if next_idx < N:
        img = Xs[next_idx]
        low, high = np.min(img), np.max(img)
        value[y0:y1, x0:x1] = ubound * (img - low) / (high - low)
        next_idx += 1
      x0 += W + padding
      x1 += W + padding
    y0 += H + padding
    y1 += H + padding
  print(value.shape)
  return value[:,:,0]