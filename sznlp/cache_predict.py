# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 16:57:33 2022

@author: Administrator
"""
from .cache_model import *
from .my_bert4keras.models import build_transformer_model
from .my_bert4keras.snippets import sequence_padding
from .my_bert4keras.tokenizers import Tokenizer


class Seq2SeqGenerate_Cache:
    # 只支持topp算法，所以进来的
    def __init__(
        self,
        encoder,  # 编码器
        decoder,  # 解码器
        tokenizer,  # 分词器
        skip_token=None,  # 换行的token，没有就是None
        start_token=None,  # 开始token，没有就用分词器默认的
        end_token=None,  # 结束token
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.skip_token = skip_token
        if start_token != None:
            self.start_token = start_token
        else:
            self.start_token = tokenizer._token_start_id

        if end_token != None:
            self.end_token = end_token
        else:
            self.end_token = tokenizer._token_end_id

    def initial_cache(self, num):
        # 初始化cache
        self.caches = [np.zeros([num, 0, t.shape[-1]]) for t in self.decoder.inputs[2:]]

    def DcoderPredict(self, encoder_outputs, decoder_out):
        # decoder模型预测
        decoder_out = np.reshape(decoder_out, [-1, 1])
        pred = self.decoder.predict(
            [encoder_outputs, decoder_out] + self.caches, batch_size=self.batch_size
        )
        self.caches = pred[1:]
        return pred[0]

    def EncoderPredict(self, encoder_inputs):
        return self.encoder.predict(encoder_inputs, batch_size=self.batch_size)

    def load_data(self, datas, nums=5):
        x = []
        for t in datas:
            x0 = self.tokenizer.encode(t)[0]
            x.extend(x0 for _ in range(nums))
        x = sequence_padding(x)
        return x

    def generate_sentence(
        self,
        data,  # 输入的独热数据
        topk=0.8,  # topk的k
        max_len=512,
        repeat_punish=0.99,
    ):
        self.initial_cache(len(data))
        encoder_outputs = self.EncoderPredict(data)
        output_ids = np.array([[self.start_token]] * len(data))
        end_token = np.array([True] * len(data))
        for step in range(max_len):
            print("\r[ nums:%d   length:%d]" % (sum(end_token), step), end="")
            decoder_out = output_ids[end_token][:, -1]
            scores = self.DcoderPredict(
                encoder_outputs[end_token], decoder_out
            )  # 计算当前得分
            scores = scores[:, -1]
            ids = output_ids[end_token]
            for i in range(len(scores)):
                for t in ids[i]:
                    if t != 0:
                        scores[i, t] *= repeat_punish
            # 这里负责解码
            outs = []
            for i in range(len(scores)):
                indices = scores[i].argsort()[::-1]  # 仅保留topk
                pre_prob = []
                pro_sum = 0
                for t in indices:
                    if pro_sum >= topk:
                        break
                    pre_prob.append(scores[i][t])
                    pro_sum += scores[i][t]
                pre_prob = np.array(pre_prob) / pro_sum
                pred = np.random.choice(indices[: len(pre_prob)], p=pre_prob)
                outs.append(pred)
            indices_2 = np.reshape(outs, (-1, 1))  # 列索引
            output_ids = np.concatenate(
                [output_ids, np.zeros([output_ids.shape[0], 1], dtype="int")], 1
            )  # 更新输出
            output_ids[end_token, -1:] = indices_2
            end_now = output_ids[end_token, -1] != self.end_token  # 标记是否以end标记结束
            end_token[end_token] = end_now

            if sum(end_token) == 0:
                break
        return output_ids

    def writer(
        self,
        data,  # 文本数据
        nums=1,  # 输入要生成几个文本
        k=0.8,
        batch_size=32,
        max_len=512,
        repeat_punish=0.99,  # 重复惩罚因子
    ):
        # 生成代码
        if k > 1 or k <= 0:
            raise ("k值应该在0和1之间")
        if repeat_punish > 1 or repeat_punish <= 0:
            raise ("惩罚值应该在0和1之间")
        data = self.load_data(data, nums)
        self.batch_size = batch_size
        ys = self.generate_sentence(data, k, max_len, repeat_punish)
        result = []

        try:
            result.extend(
                self.tokenizer.decode(y).replace(self.skip_token, "\n") for y in ys
            )

        except Exception:
            # 如果用的是SpTokenizer需要手动转成int的列表才能用
            for y in ys:
                t = [int(a) for a in y]
                result.append(self.tokenizer.decode(t).replace(self.skip_token, "\n"))
        print("\n")
        return result
