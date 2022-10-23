# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 13:22:37 2022
cpu的生成器
@author: Administrator
"""

nums=1#开头生成多个下文
k=0.8#搜索窗口
batch_size=32
max_len=512#最大长度
repeat_punish=0.99#惩罚因子
config_path='models/config-misaka.json'#config路径
vocab_path='models/vocab-misaka.txt'#词表路径
model_path='models/古言.h5'#模型路径
#开头

text='以《月色真美》为题写一篇文章'

import json
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#os.environ['TF_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sznlp.my_bert4keras.backend import set_gelu,tf,keras
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
from sznlp.my_bert4keras.models import build_transformer_model
from sznlp.cache_predict import *
def get_writer_model():
    #别动，动一下跑不了后果自负
    decoder = build_transformer_model(
                config_path=config_path,
                model=Misaka_decoder_cache,
                with_lm=True,
                return_keras_model=True,
                )
    
    encoder = build_transformer_model(
                config_path=config_path,
                model=Misaka_encoder,
                with_lm=True,
                return_keras_model=True,
                )   
    
    tokenizer=Tokenizer(vocab_path, do_lower_case=True)
    decoder.load_weights(model_path,by_name=True) 
    encoder.load_weights(model_path,by_name=True) 
    return Seq2SeqGenerate_Cache(encoder,decoder,tokenizer,skip_token='氼')


#使用方法
generate= get_writer_model() #这样子获得模型
import time
text=text.replace( '氼。','\n')
start=time.time()

#输入，建议开头字数在50字到200字之间


result=generate.writer([text.replace('\n', '氼')],#文本数据就是上面的data
               nums=nums,#输入要生成几个文本
               k=k,
               batch_size=batch_size,
               max_len=max_len,
               repeat_punish=repeat_punish)#检查重复解码
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
