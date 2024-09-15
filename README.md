# A 3D phenotyping pipeline for peanut plants using point cloud

## Segmentation Model

### Environment

```bash
sudo apt-get install libsparsehash-dev

conda create -n pointcept2 python=3.8 -y
conda activate pointcept2
conda install ninja -y
# Choose version you want here: https://pytorch.org/get-started/previous-versions/
# We use CUDA 11.8 and PyTorch 2.1.0 for our development of PTv3
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install h5py pyyaml -c anaconda -y
conda install sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm -c conda-forge -y
conda install pytorch-cluster pytorch-scatter pytorch-sparse -c pyg -y
pip install torch-geometric

cd libs/pointgroup_ops
python setup.py install
cd ../..


# PTv1 & PTv2 or precise eval
cd libs/pointops
# usual
python setup.py install

# spconv (SparseUNet)
# refer https://github.com/traveller59/spconv
pip install spconv-cu118  # choose version match your local cuda version

# Open3D (visualization, optional)
pip install open3d

# segmentator
pip install numba
cd csrc && mkdir build && cd build

cmake .. \
-DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` \
-DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())")  \
-DPYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \
-DCMAKE_INSTALL_PREFIX=`python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())'` 

make && make install 
```

FlashAttention

```bash
pip install flash-attn --no-build-isolation
```


### Dataset

training and validation datasets into the `data/huasheng3d/` folder, and modify the corresponding `configs/*` file

### Train

```bash

# Train semantic segmentaion only
python tools/train.py --config-file configs/huasheng3d/semseg-pt-v3m1-0-base.py --options save_path="exp/huasheng3d/semseg-pt-v3m1-0-base_0822"

# Train instance segmentation only
python tools/train.py --config-file configs/huasheng3d/insseg-pointgroup-v3m1-0-pt3.py --options save_path="exp/huasheng3d/insseg-pointgroup-v3m1-0-pt3_0823_inst"

# Train end to end semantic and instance segmentation
python tools/train.py --config-file configs/huasheng3d/insseg-pointgroup-v3m1-0-pt3_2.py --options save_path="exp/huasheng3d/insseg-pointgroup-v3m1-0-pt3_2_0824"
```

Our model weight is here: xxx

### Test

```bash

# Test semantic segmentaion only
python tools/test.py --config-file configs/huasheng3d/semseg-pt-v3m1-0-base.py --options save_path="exp/huasheng3d/semseg-pt-v3m1-0-base_0822" weight="exp/huasheng3d/semseg-pt-v3m1-0-base_0822/model/model_last.pth"

# Test instance segmentation only
python tools/test.py --config-file configs/huasheng3d/insseg-pointgroup-v3m1-0-pt3.py --options save_path="exp/huasheng3d/insseg-pointgroup-v3m1-0-pt3_0823_inst" weight="exp/huasheng3d/insseg-pointgroup-v3m1-0-pt3_0823_inst/model/model_last.pth"

# Test end to end semantic and instance segmentation
python tools/test.py --config-file configs/huasheng3d/insseg-pointgroup-v3m1-0-pt3_2.py --options save_path="exp/huasheng3d/insseg-pointgroup-v3m1-0-pt3_2_0824"  weight="exp/huasheng3d/insseg-pointgroup-v3m1-0-pt3_2_0824/model/model_best.pth"
```


## 3D Phenotype Measurement

```bash
#For label merging and phenotype acquisition of segmented data.
#Change the address in data_root when it is used.

#Environmental installation
conda install scipy,shapely,open3d,matplotlib,openpyxl,scikit-learn,numpy

#Calculate the plant height and spread and save it as an Excel table.
python src/PH_PW.py 

#Calculate the projected area of the plant and save it as an Excel table.
python src/PA.py 

#Calculate the plant volume and save it as an Execl table.
python src/PV.py 

#Calculate the plant spread of the plant and save it as an Execl table.
python src/PO.py 

#Calculate the upright degree of the plantand save it as an Execl table.
python src/PU.py 

#Calculate the light transmittance of the plant and save it as an Execl table.
python src/PT.py 

#Calculate the leaf length and width of the plant and save it as an Execl table.
python src/LL_LW.py 

#Calculate the leaf area of the plant and save it as an Execl table.
python src/LA.py 

#Calculate the plant leaf Angle and save it as an Execl table.
python src/LAG.py 

#It is used for merging semantic and instance tags and data.
python src/Tag_merge.py
```

# Reference

- [Pointcept](https://github.com/Pointcept/Pointcept)
- [3D Point Cloud Central Axis Aggregation Network](https://github.com/yangxin6/3D-Point-Cloud-Central-Axis-Aggregation-Network)