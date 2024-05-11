conda create -n gsdf python=3.7
conda activate gsdf
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
pip install -r requirements.txt
pip install "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"

cd submodules/NumpyMarchingCubes
python setup.py install