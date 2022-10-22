# misaka-writer-V2

## ai-续写小说

基于encoder-decoder结构的续写小说模型，https://github.com/pass-lin/misaka-writer/edit/main/README.md 介绍可以参考V1版本  
相较于V1 V2在模型上做了以下升级 
参数量 80M->200M
模型深度 8+8->14+14  
模型结构  MHA->GAU  
语言 中英双语->中文单语
环境 tf.keras->keras(tf.keras实在是太慢了，所以使用比较快的keras  
最后由于很多人只有cpu，所以给cpu写了个简单的cache优化，这样子cpu用户也能一分钟内生成结果了  
## 依赖环境

本项目的依赖有：tensorflow numpy pandas

如果使用GPU请安装 cuda 和 cudnn。

推荐的配置为 tensorflow 2.2.0/tensorflow 1.15，cuda 10.1，cudnn 7.6 keras2.3.1  

对于不支持 cuda 10 的 30 系显卡，建议使用 nvdian-tensorflow,如果实在没法用tf1.15就把#os.environ['TF_KERAS'] = '1'这个#去掉  

### 使用 conda 配置

对于 tensorflow 2.2.0：

```sh
conda create -n misaka-writer python=3.8
conda activate misaka-writer
conda install -c conda-forge pandas cudatoolkit=10.1 cudnn
pip install tensorflow==2.2.0 bert4keras numpy pandas keras==2.3.1  
```

对于 tensorflow 1.15（只限linux：

```sh
conda create -n misaka-writer python=3.8
conda activate misaka-writer
conda install -c conda-forge pandas cudatoolkit=11.2 cudnn
pip install tensorflow==2.5.0 bert4keras jiebaa
```

## 使用方法

generate_large 是gpu版本使用的，基本上用的就是V1的优化器  
generate_cache 是cpu版本使用的，写了个简单的cache优化  

`model_path` 是模型的权重路径，建议使用相对路径。

`num` 代表生成的下文的数量。 `text` 为输入，建议输入在20到250字之间。


## 训练语料

训练语料有100G中文

> 链接：https://pan.baidu.com/s/1WCiPA_tplI0AhdpDEuQ5ig <br/>
> 提取码：rlse  

## 预训练权重
目前都放在QQ群里，加群下载

## 社区

如有问题可加Q群-143626394(大群，除了本项目还有 https://github.com/BlinkDL/AI-Writer 项目群）、905398734（本项目小群），本人qq 935499957


---

老样子，misaka镇楼
  
![image](https://user-images.githubusercontent.com/62837036/170024801-1d10d8c5-266f-4ade-894c-67f30069f94f.png)
