# Reference: https://github.com/yanx27/Pointnet_Pointnet2_pytorch

import glob
import os

import numpy as np

DATA_PATH = os.path.join('data', 's3dis',
                         'Stanford3dDataset_v1.2_Aligned_Version')

with open('assets/s3dis_anno_paths.txt') as reader:
  anno_paths = [line.rstrip() for line in reader]
anno_paths = [os.path.join(DATA_PATH, p) for p in anno_paths]

with open('assets/s3dis_class_names.txt') as reader:
  g_classes = [x.rstrip() for x in reader]
g_class2label = {cls: i for i, cls in enumerate(g_classes)}

output_folder = 'data/s3dis'
os.makedirs(output_folder, exist_ok=True)


def collect_point_label(anno_path, out_filename):
  points_list = []

  for f in sorted(glob.glob(os.path.join(anno_path, '*.txt'))):
    cls = os.path.basename(f).split('_')[0]
    if cls not in g_classes:  # note: in some room there is 'staris' class...
      cls = 'clutter'
    points = np.loadtxt(f)
    labels = np.ones((points.shape[0], 1)) * g_class2label[cls]
    points_list.append(np.concatenate([points, labels], 1))  # Nx7

  data_label = np.concatenate(points_list, 0)
  xyz_min = np.amin(data_label, axis=0)[0:3]
  data_label[:, 0:3] -= xyz_min
  np.save(out_filename, data_label)


# Note: there is an extra character in the v1.2 data in Area_5/hallway_6. It's fixed manually.
for anno_path in anno_paths:
  if os.path.isdir(anno_path):
    print(anno_path)
    elements = anno_path.split('/')
    out_filename = elements[-3] + '_' + elements[
        -2] + '.npy'  # Area_1_hallway_1.npy
    collect_point_label(anno_path, os.path.join(output_folder, out_filename))
