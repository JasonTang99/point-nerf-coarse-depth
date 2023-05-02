# CSC2530 Project

## Libraries required
Run:

```pip install -r requirements.txt```

## Setup Point-NeRF
Follow the setup required in [official Point-NeRF Implementation](https://github.com/Xharlie/pointnerf).

The scripts to run our experiments are under ```pointnerf/scripts```:

* ```bash bear.sh``` - runs the bear scene using MVSNet.
* ```bash real_bear.sh``` - runs the bear scene using depth sensor initialized pointcloud.
* ```bash books.sh``` - runs the books scene using MVSNet.
* ```bash real_books.sh``` - runs the books scene using depth sensor initialized pointcloud.


## Note on running Point-NeRF experiments
The code on this repository is built on top of the [official Point-NeRF Implementation](https://github.com/Xharlie/pointnerf), which was built using [PyCUDA](https://wiki.tiker.net/PyCuda/Installation/), which is not up to date and is difficult to set up. Here are my setup details which is able to run:
* Ubuntu 22.04.2 LTS (64-bit)
* NVIDIA GeForce RTX 3060 Ti (8GB VRAM)
* NVIDIA-SMI 515.105.01
* Driver Version: 515.105.01
* CUDA Version: 11.7
* NVIDIA Cuda Compiler (nvcc) V11.7.64

