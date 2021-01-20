Ubuntu18.04

#### 1. Install Ubuntu18.04 and update source

Update to mirrors and install some softwares reference CSDN  [blog](https://blog.csdn.net/hymanjack/article/details/80285400)

#### 2.NVIDIA dirver

1. - remove old dirver

     ```
     sudo apt purge nvidia-*
     ```

2. - install dirver

     **plane one: Auto install**

     ```
     sudo ubuntu-drivers autoinstall
     ```

     **plane two: Add PPA to install the latest version of the driver**

     ```
     sudo add-apt-repository ppa:graphics-drivers/ppa
     ```

#### 3. Install CUDA and CUDANN

1. Download [cuda-10.2](https://developer.nvidia.com/cuda-10.2-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1604&target_type=runfilelocal) or other [version](https://developer.nvidia.com/cuda-toolkit-archive) 

   ```
   $ wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
   $ sudo sh cuda_10.2.89_440.33.01_linux.run
   ```

2. Download [cudann7.6.5 for CUDA 10.2](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/7.6.5.32/Production/10.2_20191118/cudnn-10.2-linux-x64-v7.6.5.32.tgz) or other [version](https://developer.nvidia.com/rdp/cudnn-archive)

   Install cudann:

   ```
   tar -axvf cudnn-10.2-linux-x64-v7.6.5.32.tgz 
   sudo cp cuda/include/cudnn.h /usr/local/cuda-10.2/include/ 
   sudo cp cuda/lib64/libcudnn* /usr/local/cuda-10.2/lib64/ 
   sudo chmod a+r /usr/local/cuda-10.2/include/cudnn.h 
   sudo chmod a+r /usr/local/cuda-10.2/lib64/libcudnn*
   
   sudo cp cuda/include/cudnn.h /usr/local/cuda/include/ 
   sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64/ 
   sudo chmod a+r /usr/local/cuda/include/cudnn.h 
   sudo chmod a+r /usr/local/cuda/lib64/libcudnn*
   ```

   See cudann version:

   ```
   cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2
   ```

3. Add environment variable

   ```
   sudo gedit ~/.bashrc
   ```

   Add to the last line of the text：

   ```
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.2/lib64
   export PATH=$PATH:/usr/local/cuda-10.2/bin
   export CUDA_HOME=$CUDA_HOME:/usr/local/cuda-10.2
   ```

   See cuda version

   ```
   nvcc -V
   ```

#### 4. Pytorch Install

​	See [Pytorch](https://pytorch.org/get-started/locally/) to choose pytorch version

