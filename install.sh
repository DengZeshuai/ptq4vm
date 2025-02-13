# 先安装 torch和torchvision
pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install -r requirements.txt

# pip install causal-conv1d==1.1.1
# install the causal-conv1d==1.1.1 from github https://github.com/Dao-AILab/causal-conv1d/releases/tag/v1.1.0
wget 'https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.1.0/causal_conv1d-1.1.0+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl' -O causal_conv1d-1.1.0+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl


# install the wheel from github https://github.com/state-spaces/mamba/releases/tag/v1.2.0.post1
wget 'https://objects.githubusercontent.com/github-production-release-asset-2e65be/725839295/bc4f8461-39ac-4c80-bb8c-cacde2ceac06?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=releaseassetproduction%2F20250201%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250201T144155Z&X-Amz-Expires=300&X-Amz-Signature=3640818379c29a4bf74f629f7cdd4c29d73485b82450e106fded25a6b7fe105c&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3Dmamba_ssm-1.2.0.post1%2Bcu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl&response-content-type=application%2Foctet-stream' -O mamba_ssm-1.2.0.post1+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# for huangpu server
# cp -rf mamba-1p1p1/mamba_ssm ~/miniconda3/envs/ptq4vm/lib/python3.10/site-packages/
# for pazhoulab server
cp -rf mamba-1p1p1/mamba_ssm /root/miniconda3/envs/ptq4vm/lib/python3.10/site-packages

# cuda 驱动以满足编译要求

# conda install nvidia/label/cuda-11.8.0::cuda
conda install cudatoolkit=11.8
# conda install cudatoolkit-dev=11.8 -c conda-forge
conda install senyan.dev::cudatoolkit-dev # 11.8
# conda install rocketce/label/rocketce-1.9.1::cudatoolkit-dev # 11.8

# set the CUDA_HOME to the conda prefix
export CUDA_HOME=$CONDA_PREFIX 

# 编译安装
# change the directory in the setup_vim_GEMM.py from vim_GEMM.cpp to ./vim_GEMM.cpp
# change the directory in the setup_vim_GEMM.py from vim_GEMM_kernel.cu to ./vim_GEMM_kernel.cu

# git clone https://github.com/NVIDIA/cutlass.git
cd ./cuda_measure/ $$  python setup_vim_GEMM.py install
# ln -s build/ ../ 
# python ./cuda_measure/setup_vim_GEMM.py install
