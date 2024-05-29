# 1 安装环境
先装pytorch根据需求安装版本，我装的是1.12.1

```zsh
pip install ultralytics
```
服务器已有环境为 ubuntu 用户下的yolov8 conda 环境。
```zsh
su ubuntu
zsh
conda activate yolov8
```
# 2 数据集

服务器数据集文件路径
```zsh
/home/ubuntu/DataSet/BS
```

# 3 运行训练

```zsh
python train.py
```

修改超参数可以直接修改配置文件里的参数
```zsh
/home/ubuntu/GITHUB/ultralytics/ultralytics/cfg/default.yaml
```

# 4 normalize 文件位置
```
/home/ubuntu/DataSet/FULL_middle_eric/convert_png_normalize.py
```