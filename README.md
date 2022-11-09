# misaka-writer-V2

## ai-续写小说

基于 encoder-decoder 结构的续写小说模型，可以参考[V1 版本的介绍](https://github.com/pass-lin/misaka-writer/blob/main/README.md)。

相较于 V1，V2 在模型上做了以下升级：
| | |
|--|--|
|参数量|80M->200M|
|模型深度| 8+8->15+15 |
|模型结构 | MHA+FFN->GAU+GAU|
|语言 |中英双语->中文单语|
|环境 |tf.keras->keras(tf.keras 实在是太慢了，所以使用比较快的 keras |

最后，由于很多人只有 cpu，所以给 cpu 写了个简单的 cache 优化，这样子 cpu 用户也能一分钟内生成结果了。

## 依赖环境

本项目的依赖有：`tensorflow` `numpy` `pandas` `sklearn`。

如果使用 GPU 请安装 cuda 和 cudnn。

推荐的配置为 tensorflow 2.2.0/tensorflow 1.15，cuda 10.1，cudnn 7.6，keras2.3.1。

对于不支持 cuda 10 的 30 系显卡，建议使用 nvdia-tensorflow 或 tensorflow-directml，或者可以设置环境变量 `TF_KERAS` 为 1 来支持高版本的 tensorflow。

### 使用 conda 配置

对于 tensorflow 2.2.0：

```sh
conda create -n misaka-writer python=3.8
conda activate misaka-writer
conda install -c conda-forge pandas cudatoolkit=10.1 cudnn
pip install tensorflow==2.2.0 keras==2.3.1 sklearn
```

对于 tensorflow 1.15（只限 linux）：

```sh
conda create -n misaka-writer python=3.8
conda activate misaka-writer
conda install -c conda-forge pandas cudatoolkit=10.1 cudnn
pip install tensorflow-gpu==1.15.0 keras==2.3.1 sklearn 
```

对于 tensorflow-directml（只限 Windows 10/11 或 wsl，此版本不需要安装 CUDA）：

```sh
conda create -n misaka-writer python=3.7
conda activate misaka-writer
conda install -c conda-forge pandas
pip install tensorflow-directml keras==2.3.1 sklearn
```

## 使用方法

见 `generate.py`，基本上用的是 V1 的优化器，此外对 cpu 写了简单的 cache 优化。

`model_path` 是模型的权重路径。

`num` 代表生成的下文的数量。 `text` 为输入，建议输入在 20 到 250 字之间。

## WebUI

使用 `streamlit` 启动 WebUI，见 `webui.py`。

```sh
pip install streamlit>=1.10.0
streamlit run webui.py
```

## 训练语料

训练语料有 100G 中文：

> 链接：https://pan.baidu.com/s/1WCiPA_tplI0AhdpDEuQ5ig <br/>
> 提取码：rlse

## 预训练权重

目前都放在 QQ 群里，加群下载。

## 社区

如有问题可加 Q 群-143626394(大群，除了本项目还有 https://github.com/BlinkDL/AI-Writer 项目群）、905398734（本项目小群），本人 qq 935499957

---

最后用ai生成的misaka镇楼  
![QQ图片20221109142639](https://user-images.githubusercontent.com/62837036/200754613-febeb470-7e27-4347-9b31-340e090b87ab.png)

