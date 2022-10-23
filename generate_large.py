# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 13:22:37 2022
gpu的生成器
@author: Administrator
"""
model_path='models/古言.h5'#保存路径
nums=1#开头生成多个下文
config_path='models/config-misaka.json'#config路径
vocab_path='models/vocab-misaka.txt'#词表路径
#开头

text='''白月耸了耸肩膀，无语。黎傲然不再说话，继续闭上眼睛养起神来。
　　马车慢慢的驶出了城，在城外宽阔的大道上前行着。
　　“咦？”凌言看着窗外越来越僻静的小路，觉察出了不对劲。
　　“怎么了？”白月不解的看着凌言。
　　“似乎方向不对啊。”凌言将头探出窗外，大声冲车夫道，“师傅，你是不是走错路了？”
　　车夫却丝毫不理会凌言的话，反而扬起鞭子抽了马一鞭子，将马车赶的更快了。
  '''

import json
import os

#os.environ['TF_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sznlp.my_bert4keras.backend import set_gelu,tf,keras
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
from sznlp.my_bert4keras.tokenizers import Tokenizer
from sznlp.tools import seq2seq_Generate
from sznlp.misaka_models import Misaka
from sznlp.my_bert4keras.models import build_transformer_model
def get_writer_model():
    #别动，动一下跑不了后果自负
    misaka = build_transformer_model(
            config_path=config_path,
            model=Misaka,
            with_lm=True,
            return_keras_model=False,
            )
    misaka.model.summary()
    
    tokenizer=Tokenizer(vocab_path, do_lower_case=True)
    misaka.model.load_weights(model_path,by_name=True) 
    encoder=misaka.encoder
    decoder=misaka.decoder
    outputs = [
                keras.layers.Lambda(lambda x: x[:, -1:])(output)
                for output in decoder.outputs
            ]
    decoder = keras.models.Model(decoder.inputs, outputs)
    return seq2seq_Generate(encoder,decoder,tokenizer)


#使用方法
generate= get_writer_model() #这样子获得模型
import time
start=time.time()

#输入，建议开头字数在50字到200字之间


result=generate.writer([text.replace('\n', '氼')],#文本数据就是上面的data
               nums=nums,#一个开头要生成几个文本
               k=0.8,#搜索窗口
               batch_size=32,
               max_len=1500,#最大长度
               iter_data_num=400,#一次处理多少个开头
               mode='topp',#别动的句子的次数，越大就越慢同时重复句子越少)
               iter_max_num=0,)#检查重复解码
end=time.time()
s=''
for t in text.split('\n'):
    s+='\t'+t+'\n'
text=s
for i in range(nums):
    print(text)
    print('*******************************************************************************')
    for t in result[i].split('氼'):
        print('\t'+t)
    print('*******************************************************************************')
print('消耗时间'+str(end-start))
