import cv2

import io
import numpy as np
import pathlib
import time
from math import hypot

size = (128, 128)
harf_area = 5.0
map_resolution = 10.0 / 128.0

def occupy_grid(radius):
  radius_up = radius
  grid = int(radius_up / map_resolution)
  if radius_up > grid * map_resolution:
    grid = grid + 1
    radius_up = grid * map_resolution

  occ_size = grid * 2 + 1
  occupation = np.zeros((occ_size, occ_size), dtype=np.uint8)
  for i in range(grid + 1):
    for j in range(grid + 1):
      dy = i * map_resolution
      dx = j * map_resolution
      dis = hypot(dx, dy)
      if dis <= radius_up:
        occupation[grid - i][grid - j] = 255
        occupation[grid + i][grid - j] = 255
        occupation[grid + i][grid + j] = 255
        occupation[grid - i][grid + j] = 255
      else:
        break
  return occupation, grid

def position_to_map(px, py):
  # Cartesian coordinate to pixel coordinate
  # Cartesian frame
  """
                            ^ X axis
                            |
                            |
                            |
                            |
                            |
                            |
                            |
                            |
  <-------------------------
  Y axis
  """
  map_h = int((harf_area - px) / map_resolution)
  map_w = int((harf_area - py) / map_resolution)
  return map_h, map_w

def inflation_area(px, py, grid_radius):
  map_h, map_w = position_to_map(px, py)

  map_up = map_h - grid_radius
  if map_up < 0:
    up = 0
    up_inflation = -map_up
  else:
    up = map_up
    up_inflation = 0

  map_down = map_h + grid_radius
  if map_down >= size[0]:
    down = size[0] - 1
    down_inflation = 2 * grid_radius - (map_down - down)
  else:
    down = map_down
    down_inflation = 2 * grid_radius

  map_left = map_w - grid_radius
  if map_left < 0:
    left = 0
    left_inflation = -map_left
  else:
    left = map_left
    left_inflation = 0

  map_right = map_w + grid_radius
  if map_right >= size[0]:
    right = size[0] - 1
    right_inflation = 2 * grid_radius - (map_right - right)
  else:
    right = map_right
    right_inflation = 2 * grid_radius
  return up, up_inflation, down, down_inflation, left, left_inflation, right, right_inflation


directory = './logdir/online/5/eval_episodes'
directory_path = pathlib.Path(directory).expanduser()

goal_grid_area, goal_grid_raidus = occupy_grid(0.3)


goal_map = np.zeros((size[0], size[1], 1), dtype=np.uint8)
up, up_inflation, down, down_inflation, \
  left, left_inflation, right, right_inflation = inflation_area(4.0, 0.0, goal_grid_raidus)
goal_map[up:down+1, left:right+1, 0] = \
  goal_grid_area[up_inflation:down_inflation+1, left_inflation:right_inflation+1]


ep = 0

for filename in reversed(sorted(directory_path.glob('*.npz'))):
  try:
    with filename.open('rb') as f:
      episode = np.load(f)
      episode = {k: episode[k] for k in episode.keys()}
  except Exception as e:
    print(f'Could not load episode: {e}')
    continue
  ep_str = '/video_' + str(ep) + '.mp4'
  video_fn = directory + ep_str
  # print(episode['reward'][:100])
  out = cv2.VideoWriter(video_fn, cv2.VideoWriter_fourcc(*'mp4v'), 5, size)
  for idx in range(episode['image'].shape[0]):
    image_original = episode['image'][idx]
    for i in range(size[0]):
      for j in range(size[0]):
        image_original[i, j, 1] = goal_map[i][j]
    # cv2.namedWindow('video', cv2.WINDOW_AUTOSIZE)
    # cv2.imshow('video', image_original)
    # time.sleep(0.2)
    # cv2.waitKey(1)

    out.write(image_original)
    
    
    
  out.release()
  ep += 1