# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 15:05:31 2022

@author: Administrator
"""
import time
from sklearn.utils import shuffle
import numpy as np
from my_bert4keras.snippets import sequence_padding
from my_bert4keras.backend import keras,tf,K
from  threading import Thread
import random
import os
datas=[]
def sample_data(filenames,funtion,max_data_num=2):
    global datas
    while True:
        filenames=shuffle(filenames)
        for filename in filenames:        
            while len(datas)>max_data_num:
                time.sleep(1)
            try:
                datas.append(funtion(filename))
            except:
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print('find error in '+filename)
                continue

class DataGenerator(object):
    """数据生成器模版
    """
    def __init__(self, file_dir,load_function,max_data_num=5,thread_num=3):
        self.datas=[]
        self.filenames=[]
        filenames= os.listdir(file_dir)
        for i in range(len(filenames)):
            if '.csv' in filenames[i]:
                self.filenames.append(file_dir+filenames[i])
        self.load_function=load_function
        self.max_data_num=max_data_num
        self.threads =[]
        for _ in range(thread_num):
            thread=Thread(target=sample_data,args=(self.filenames,self.load_function,max_data_num))
            thread.start()
            self.threads.append(thread)

    def batch_compute(self,datas):
        return self.batch_size
    def sample_datas(self,sample_data,random=False):
        pass
    def __iter__(self,random=False):
        global datas
        while len(datas)==0:
            print('wait data')
            time.sleep(15)
        self.data=datas.pop(0)
        for d in self.sample_datas(self.data,random):
            yield d
    def forfit(self, random=False):
        n=1
        while True:
            
            for d in self.__iter__(random):
                yield d
            print('完成第'+str(n)+'个文件的训练')
            n+=1
class seq2seq_Generate():
    def __init__(self,encoder,#编码器
                 decoder,#解码器
                 tokenizer,#分词器
                 start_token=None,#开始token，没有就用分词器默认的
                 end_token=None#结束token
                 ):
        self.encoder=encoder
        self.decoder=decoder
        self.tokenizer=tokenizer
        if start_token!=None:
            self.start_token=start_token
        else:
            self.start_token=tokenizer._token_start_id
            
        if end_token!=None:
            self.end_token=end_token
        else:
            self.end_token=tokenizer._token_end_id
    def encoder_predict(self,data,batch_size=32):
        #编码
        return self.encoder.predict(data,batch_size=batch_size)
    def decoder_predict(self,encoder_out,decoder_input,batch_size=32):
        #解码
         return self.decoder.predict([encoder_out,decoder_input],batch_size=batch_size)
    def select(self,encoder_out,decode_result,index):
        #筛选
        decoder_input=decode_result[index]
        vector=encoder_out[index]
        return decoder_input,vector
    def greedy_search(self,data,k=10,batch_size=32,max_len=512):
        encoder_out=self.encoder_predict(data,batch_size)
        l=len(data)
        decode_result=np.array([[self.start_token]]*l)#初始数据
        n=0
        stop=np.zeros(l)
        while sum(stop[:]==self.end_token)!=l:
            if n>max_len:
                break
            n+=1
            index=stop[:]!=self.end_token
            decoder_input,vector=self.select(encoder_out,decode_result,index)
            y=self.decoder_predict(vector,decoder_input,batch_size)
            y=np.argmax(y,-1)#选取最高值
            stop[index]=y[:,-1]
            #生成新的对齐数据
            t=np.zeros([l])
            t[index]=y[:,-1]
            t=np.reshape(t,[l,1])
            decode_result=np.concatenate([decode_result,t],-1)
        return decode_result,stop
    def topk_sample(self,y,k=10):
        y=y[:,-1,:]
        #y=K.softmax(y,-1)#先计算真实预测分布
        
        probility=np.sort(y,axis=-1)[:,-k:]
        
        probility_s=np.sum(probility,axis=-1,keepdims=True)
        probility=probility/probility_s#归一化
        #选取最搞高的k个
        y_sort=np.argsort(y,-1)[:,-k:]#这里用numpy是方便转array
        #根据概率采样
        y=[np.random.choice(y_sort[i],p=probility[i]) for i in range(len(y_sort))]
        y=np.reshape(y,[-1,1])
        return y
    def topp_sample(self,y,k=0.8):
        y=y[:,-1,:]
        #y=K.softmax(y,-1)#先计算真实预测分布
        
        probility=np.argsort(y,axis=-1)
        
        y_pre=[]
        for i in range(len(y)):
            pre=[probility[i][-1]]
            pro=[y[i][probility[i][-1]]]
            sump=y[i][probility[i][-1]]
            for j in range(2,len(probility[0])):
                j=j*-1
                if sump>k:
                    break
                pre.append(probility[i][j])
                pro.append(y[i][probility[i][j]])
                sump+=y[i][probility[i][j]]
            pro=np.array(pro)
            pro/=sum(pro)
            y_pre.append(np.random.choice(pre,p=pro))
        y=np.reshape(y_pre,[-1,1])
        return y
    def random_decoder(self,data,k=10,batch_size=32,max_len=512,mode='topk'):
        l=len(data)
        if mode=='topk':
            sample=self.topk_sample
        elif mode=='topp':
            if k>1:
                raise('topp的k值应该为0到1之间')
            sample=self.topp_sample
        decode_result=np.array([[self.start_token]]*l)
        n=0
        encoder_out=self.encoder_predict(data,batch_size)
        stop=np.zeros(l)
        while sum(stop[:]==self.end_token)!=l:#终止运算条件一全都预测完
            if n>max_len:#二预测到最大长度
                break
            n+=1
            
            index=stop[:]!=self.end_token#找出预测完的序列
            
            print('\r [nums:'+str(sum(stop[:]!=self.end_token))+' length:'+str(n)+']', end="",flush = True)
            #只预测没预测完的序列以节省资源
            decoder_input,vector=self.select(encoder_out,decode_result,index)
            #解码
            y=self.decoder_predict(vector,decoder_input,batch_size)
            #计算概率分布
            y=sample(y,k)
            
            stop[index]=y[:,-1]
            t=np.zeros([l])
            t[index]=y[:,-1]
            t=np.reshape(t,[l,1])
            decode_result=np.concatenate([decode_result,t],-1)
        return decode_result,stop
    def load_data(self,datas,nums=5):
        x=[]
        for t in datas:
            x0=self.tokenizer.encode(t)[0]
            for _ in range(nums):
                x.append(x0)
        x=sequence_padding(x)
        return x
    def copy(self,old,stop,new,newstop):
        length1=old.shape[-1]
        length2=new.shape[-1]
        if length1>length2:
            news=np.pad(new,((0,0),(0,length1-length2)),'constant',constant_values=(1,1))
            result=old
        else:
            news=new
            result=np.pad(old,((0,0),(0,length2-length1)),'constant',constant_values=(1,1))
        result[stop[:]!=self.end_token]=news[:]
        stop[stop[:]!=self.end_token]=newstop[:]
        return   result,stop
    def predict(self,data,k=10,batch_size=32,max_len=512,iter_data_num=400,iter_max_num=1,mode='topk'):
        if mode=='greedy':
            a,stop=self.greedy_search(data,k,batch_size,max_len)
            
        else:
            a,stop=self.random_decoder(data,k,batch_size,max_len,mode=mode)
        #如果存在重复解码现象，那就再随机解码一次
        while sum(stop[:]==self.end_token)!=len(stop) and iter_max_num>0:
            t=self.random_decoder(data[stop[:]!=self.end_token],k,batch_size,max_len,mode)
            a,stop=self.copy(a,stop,t[0],t[1])
            if iter_max_num>0:
                iter_max_num-=1
            else:
                break
        return list(a)
        
    def generate_sentence(self,data,#输入的独热数据
                          k=10,#topk的k
                          batch_size=32,
                          max_len=512,
                          iter_data_num=400,#一次迭代迭代多少数据
                          mode='topk',#计算模式
                          iter_max_num=1,
                          ):
        result=[]
        l=len(data)
        #因为显存限制，数据过大的时候需要分散解码
        if l<iter_data_num:
            return self.predict(data,k,batch_size,max_len,mode,iter_max_num=iter_max_num,mode=mode)
        else:  
            for i in range(l//iter_data_num):
                j=i*iter_data_num
                result.extend(self.predict(data[j:j+iter_data_num],k,batch_size,max_len,mode))
        return result
    def writer(self,data,#文本数据
               nums=1,#输入要生成几个文本
               k=10,
               batch_size=32,
               max_len=512,
               iter_data_num=400,
               mode='topk',
               iter_max_num=1,):
        data=self.load_data(data,nums)
        ys=self.generate_sentence(data,k,batch_size,max_len,iter_data_num,mode,iter_max_num=iter_max_num)
        result=[]
        try:
            for y in ys:
                result.append(self.tokenizer.decode(y))
        except:
            #如果用的是SpTokenizer需要手动转成int的列表才能用
            for y in ys:
                
                t=[]
                for a in y:
                    t.append(int(a))
                result.append(self.tokenizer.decode(t))
        return result