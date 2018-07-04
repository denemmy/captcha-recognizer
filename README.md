## Requirements
Code is written in python3, so you should have it
- Install [cuda](https://developer.nvidia.com/cuda-downloads)
- Install [cudnn](https://developer.nvidia.com/cudnn)
- Install [opencv](https://pypi.org/project/opencv-python/)
- Install [mxnet](https://mxnet.incubator.apache.org/install/index.html?platform=Linux&language=Python&processor=GPU)

```bash
pip3 install opencv-python
pip3 install mxnet-cu80 (CUDA 8.0)
git clone https://github.com/denemmy/captcha-recognizer.git
```

## How to run
After you have cloned the repository, you can test model running the script below.
```bash
cd deploy
python3 deploy.py --gpu 0 --input <directory with images>
```
This script create `results.txt` file in the input directory with predicted labels for each image.
