if [ ! -d "data/semantic3d" ]; then
  echo "Preparing Semantic3D dataset..."
  if [ -f "data/birdfountain_station1_xyz_intensity_rgb.7z" ]; then
    7za x -y data/birdfountain_station1_xyz_intensity_rgb.7z -odata/semantic3d
  else
    echo "Semantic3D dataset is not downloaded at data/birdfountain_station1_xyz_intensity_rgb.7z"
  fi
fi

if [ ! -d "data/s3dis" ]; then
  echo "Preparing S3DIS dataset..."
  if [ -f "data/Stanford3dDataset_v1.2_Aligned_Version.zip" ]; then
    unzip data/Stanford3dDataset_v1.2_Aligned_Version.zip "Stanford3dDataset_v1.2_Aligned_Version/Area_5/*" -d data/s3dis
    patch -i assets/s3dis_patch.txt data/s3dis/Stanford3dDataset_v1.2_Aligned_Version/Area_5/hallway_6/Annotations/ceiling_1.txt
    python3 scripts/prepare_s3dis.py
  else
    echo "S3DIS dataset is not downloaded at data/Stanford3dDataset_v1.2_Aligned_Version.zip"
  fi
fi

if [ ! -d "data/semantic_kitti" ]; then
  echo "Preparing SemanticKITTI dataset..."
  if [ -f "data/data_odometry_velodyne.zip" ]; then
    unzip data/data_odometry_velodyne.zip "dataset/sequences/08/*" -d data/semantic_kitti
  else
    echo "SemanticKITTI dataset is not downloaded at data/data_odometry_velodyne.zip"
  fi
fi

if [ ! -d "data/shapenet" ]; then
  echo "Preparing ShapeNetSem dataset..."
  if [ -f "data/ShapeNetSem.zip" ]; then
    unzip data/ShapeNetSem.zip "ShapeNetSem-backup/models-binvox/*" -d data/shapenet
  else
    echo "ShapeNetSem dataset is not downloaded at data/ShapeNetSem.zip"
  fi
fi
