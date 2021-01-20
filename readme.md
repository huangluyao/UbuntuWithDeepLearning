[Ubuntu 18.04 with DeepLearning (cuda10.2 + cudnn8.0 + TensorRT-7.2+ OnnxRuntime1.3)](# # Ubuntu 18.04 with DeepLearning (cuda10.2 + cudnn8.0 + TensorRT-7.2+ OnnxRuntime1.3))
* [1. Install Ubuntu18.04 and update source](## 1. Install Ubuntu18.04 and update source)
* [2.NVIDIA dirver](## 2.NVIDIA dirver)
* [3. Install CUDA and CUDANN](## 3. Install CUDA and CUDANN)
* [4. Pytorch Install](## 4. Pytorch Install)
* [5.Install TensorRT](## 5.Install TensorRT)
* [6. Install OnnxRuntime](## 6. Install OnnxRuntime)
# Ubuntu 18.04 with DeepLearning (cuda10.2 + cudnn8.0 + TensorRT-7.2+ OnnxRuntime1.3)

## 1. Install Ubuntu18.04 and update source

Update to mirrors and install some softwares reference CSDN  [blog](https://blog.csdn.net/hymanjack/article/details/80285400)

## 2.NVIDIA dirver

1. remove old dirver

   ```
   sudo apt purge nvidia-*
   ```

2. install dirver

   ```
   $ sudo add-apt-repository ppa:graphics-drivers/ppa
   $ sudo apt update
   $ sudo ubuntu-drivers autoinstall
   ```

3. Check 

   ```
   nvidia-smi
   ```

## 3. Install CUDA and CUDANN

1. Download [cuda-10.2](https://developer.nvidia.com/cuda-10.2-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1604&target_type=runfilelocal) or other [version](https://developer.nvidia.com/cuda-toolkit-archive) 

   ```
   $ wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
   $ sudo sh cuda_10.2.89_440.33.01_linux.run
   ```

2. Download [cudann-8.02 for CUDA 10.2](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.0.2.39/10.2_20200724/cudnn-10.2-linux-x64-v8.0.2.39.tgz) or other [version](https://developer.nvidia.com/rdp/cudnn-archive)

   Copy the following files into the CUDA Toolkit directory:

   ```
   $ tar -xzvf cudnn-x.x-linux-x64-v8.x.x.x.tgz
   $ sudo cp cuda/include/cudnn*.h /usr/local/cuda/include
   $ sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
   $ sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
   ```
   
   See cudann version:
   
   ```
   cat /usr/local/cuda/include/cudnn_cnn_infer.h | grep CUDNN_CNN_INFER_MAJOR -A 2
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

## 4. Pytorch Install

See [Pytorch](https://pytorch.org/get-started/locally/) to choice pytorch version

## 5.Install TensorRT

1. Download [tensorrt7.2.1 for cuda 10.2](https://developer.nvidia.com/compute/machine-learning/tensorrt/secure/7.2.1/tars/TensorRT-7.2.1.6.Ubuntu-18.04.x86_64-gnu.cuda-10.2.cudnn8.0.tar.gz) or other [version](https://developer.nvidia.com/nvidia-tensorrt-7x-download):

```
tar -xzvf TensorRT-7.2.1.6.Ubuntu-18.04.x86_64-gnu.cuda-10.2.cudnn8.0.tar.gz
```

2. Installation package

```
pip install python/tensorrt-7.2.1.6-cp36-none-linux_x86_64.whl 
pip install uff/uff-0.6.9-py2.py3-none-any.whl
pip install onnx_graphsurgeon/onnx_graphsurgeon-0.2.6-py2.py3-none-any.whl 
```

 3. Test mnist_sample by C++

    Compile sample program:

    ```
    $ cd sample
    $ make clean
    $ make
    ```

    Get mnist data:

    ```
    $ cd ../data/mnist
    $ python download_pgms.py
    ```

    Add environment variable:

    ```
    $ sudo gedit ~/.bashrc
    ```

    ```
    export LD_LIBRARY_PATH=$LD_LIRARY_PATH:{TensrRT Path}/TensorRT-7.2.1.6/lib
    ```

    Restart Terminal and under folder TensorRT-7.2.1.6/bin to use

    ```
    ./sample_mnist
    ```

## 6. Install OnnxRuntime

Download source

```
git clone -b rel-1.3.0 --recursive https://github.com/Microsoft/onnxruntime
```

or use mirror:

```
git clone -b rel-1.3.0 --recursive https://github.com.cnpmjs.org/Microsoft/onnxruntime
```

Update package

```
cd onnxruntime && git submodule update --init --recursive
```

Install

```
./build.sh --build_shared_lib --config Release --use_cuda --cudnn_home /usr/local/cuda-10.0/ --cuda_home /usr/local/cuda-10.0/ --use_tensorrt --tensorrt_home /home/hly/workspace/library/TensorRT-7.2.1.6 --update --build
```

