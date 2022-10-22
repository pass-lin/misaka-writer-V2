# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 14:29:15 2022

@author: Administrator
"""

from .misaka_models import *


class Add(Layer):
    def call(self, inputs):
        return inputs[0] + inputs[1]


class GatedAttentionUnit_cross_cache(GatedAttentionUnit_cross):
    def call(self, inputs, mask=None, a_bias=False, p_bias=None, **kwargs):
        if not isinstance(inputs, list):
            inputs, mask = [inputs], [mask]
        x, c = inputs[:]
        n = 1

        if a_bias:
            a_bias = inputs[n]
            n += 1
        # 投影变换
        x = self.uq_dense(x)
        u, q = tf.split(x, [self.units, self.key_size], axis=-1)
        if (
            not K.int_shape(c)[1]
            or K.int_shape(kwargs.get("c_cache"))[1] != K.int_shape(c)[1]
        ):
            c = self.kv_dense(c)

        else:
            c = kwargs.get("c_cache")

        # 加入RoPE
        v, k = tf.split(c, [self.units, self.key_size], axis=-1)
        if p_bias == "rotary":
            q, k = apply_rotary_position_embeddings(inputs[n], q, k)
        # Attention
        a = tf.einsum("bmd,bnd->bmn", q, k)
        if self.attention_scale:
            a = a / self.key_size**0.5
        if a_bias is not None:
            a = a + a_bias

        A = attention_normalize(a, -1, self.normalization)
        if self.attention_dropout:
            A = Dropout(self.attention_dropout)(A)
        # 计算输出
        o = self.o_dense(u * tf.einsum("bmn,bnd->bmd", A, v))
        return [o, c]

    def compute_output_shape(self, input_shape):
        o_shape = (input_shape[0][0], input_shape[0][1], input_shape[0][-1])
        kw_shape = (input_shape[1][0], input_shape[1][1], self.key_size + self.units)
        vw_shape = (input_shape[1][0], input_shape[1][1], self.units)
        return [o_shape, kw_shape]


class GatedAttentionUnit_cache(GatedAttentionUnit):
    def call(self, inputs, mask=None, a_bias=None, p_bias=None, **kwargs):
        if not isinstance(inputs, list):
            inputs, mask = [inputs], [mask]
        x, n = inputs[0], 1
        mask = None if mask is None else mask[0]
        if a_bias:
            a_bias = inputs[n]
            n += 1
        # 投影变换
        x = self.i_dense(x)
        u, v, qk = tf.split(x, [self.units, self.units, self.key_size], axis=-1)

        q, k = self.q_scaleoffset(qk), self.k_scaleoffset(qk)

        # 加入RoPE
        if p_bias == "rotary":
            q, k = apply_rotary_position_embeddings(inputs[n], q, k)

        k = tf.concat([kwargs.get("k_cache"), k], axis=1)
        v = tf.concat([kwargs.get("v_cache"), v], axis=1)

        # Attention
        a = tf.einsum("bmd,bnd->bmn", q, k)
        if self.attention_scale:
            a = a / self.key_size**0.5
        if a_bias is not None:
            a = a + a_bias

        A = attention_normalize(a, -1, self.normalization)
        if self.attention_dropout:
            A = Dropout(self.attention_dropout)(A)
        # 计算输出
        o = self.o_dense(u * tf.einsum("bmn,bnd->bmd", A, v))
        return [o, k, v]

    def compute_output_shape(self, input_shape):
        o_shape = (input_shape[0], input_shape[1], input_shape[-1])
        kw_shape = (input_shape[0], input_shape[1], self.key_size)
        vw_shape = (input_shape[0], input_shape[1], self.units)
        return [o_shape, kw_shape, vw_shape]


class SinusoidalPositionEmbedding_cache(SinusoidalPositionEmbedding):
    """定义Sin-Cos位置Embedding"""

    def __init__(self, max_len=512, **kwargs):
        super(SinusoidalPositionEmbedding_cache, self).__init__(**kwargs)
        self.max_len = max_len

    def build(self, input_shape):
        super(SinusoidalPositionEmbedding_cache, self).build(input_shape)
        position_ids = K.arange(0, self.max_len, dtype=K.floatx())[None]
        indices = K.arange(0, self.output_dim // 2, dtype=K.floatx())
        indices = K.pow(10000.0, -2 * indices / self.output_dim)
        embeddings = tf.einsum("bn,d->bnd", position_ids, indices)
        embeddings = K.stack([K.sin(embeddings), K.cos(embeddings)], axis=-1)
        embeddings = K.flatten(embeddings, 2)

        self.embeddings = embeddings[0]

    def call(self, inputs):
        k_cache = inputs
        seq_len = k_cache.shape[1]
        if seq_len is not None and seq_len >= self.max_len:
            indices = K.arange(0, self.output_dim // 2, dtype=K.floatx())
            indices = K.pow(10000.0, -2 * indices / self.output_dim)
            embeddings = tf.einsum("bn,d->bnd", tf.ones([1, 1]) * seq_len, indices)
            embeddings = K.stack([K.sin(embeddings), K.cos(embeddings)], axis=-1)
            embeddings = K.flatten(embeddings, 2)
        else:

            embeddings = self.embeddings[: K.shape(k_cache)[1] + 1][-1]
            embeddings = K.reshape(embeddings, [1, 1, self.output_dim])
        return embeddings


class Misaka_decoder_cache(Misaka_decoder):
    def __init__(self, maxlength=512, **kwargs):
        super(Misaka_decoder_cache, self).__init__(**kwargs)
        self.cache_inputs = []
        self.cache_outputs = []
        self.cache_position = []
        self.maxlength = maxlength

    def compute_position_bias(self, inputs=None):
        """Sinusoidal位置编码（直接返回）"""
        if self.position_bias is None:

            self.position_bias = self.apply(
                inputs=inputs,
                layer=SinusoidalPositionEmbedding_cache,
                output_dim=self.attention_key_size,
                merge_mode="zero",
                max_len=self.maxlength,
                name="Embedding-Rotary-Position",
            )

        return self.position_bias

    def apply_main_layers(self, inputs, index):
        """Misaka-encoder 的主体是基于Gated Attention Unit的模块
        顺序：GAU  --> Add --> LN
        """
        c, x = inputs

        self_attention_1_name = "Misaka-Dncoder-%d-GatedAttentionUnit-1" % index
        k_cache_1 = keras.layers.Input(
            [None, self.attention_key_size], name=self_attention_1_name + "-kcache"
        )
        self.cache_inputs.append(k_cache_1)
        v_cache_1 = keras.layers.Input(
            [None, self.intermediate_size], name=self_attention_1_name + "-vcache"
        )
        self.cache_inputs.append(v_cache_1)

        cross_attention_name = "Misaka-Dncoder-%d-GatedAttentionUnit-cross" % index

        c_cache_cross = keras.layers.Input(
            [None, self.intermediate_size + self.attention_key_size],
            name=cross_attention_name + "-cache",
        )
        self.cache_inputs.append(c_cache_cross)

        self_attention_2_name = "Misaka-Dncoder-%d-GatedAttentionUnit-2" % index
        k_cache_2 = keras.layers.Input(
            [None, self.attention_key_size], name=self_attention_2_name + "-kcache"
        )
        self.cache_inputs.append(k_cache_2)
        v_cache_2 = keras.layers.Input(
            [None, self.intermediate_size], name=self_attention_2_name + "-vcache"
        )
        self.cache_inputs.append(v_cache_2)

        attention_mask = None  # self.compute_attention_bias(index)
        position_bias = self.compute_position_bias(k_cache_1)

        # GAU-1
        xi = x
        x = [x, position_bias]
        arguments = {"a_bias": None, "p_bias": "rotary"}
        if attention_mask is not None:
            arguments["a_bias"] = True
            x.insert(1, attention_mask)

        x, k, v = self.apply(
            inputs=x,
            layer=GatedAttentionUnit_cache,
            units=self.intermediate_size,
            key_size=self.attention_key_size,
            activation=self.hidden_act,
            use_bias=False,
            normalization="softmax_plus",
            attention_dropout=self.attention_dropout_rate,
            kernel_initializer=self.initializer,
            name=self_attention_1_name,
            arguments={
                "a_bias": arguments["a_bias"],
                "p_bias": arguments["p_bias"],
                "k_cache": k_cache_1,
                "v_cache": v_cache_1,
            },
        )
        self.cache_outputs.append(k)
        self.cache_outputs.append(v)

        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name="%s-Dropout" % self_attention_1_name,
        )
        x = self.apply(inputs=[xi, x], layer=Add, name="%s-Add" % self_attention_1_name)
        x = self.apply(
            inputs=x,
            layer=LayerNormalization,
            zero_mean=False,
            scale=False,
            offset=False,
            name="%s-Norm" % self_attention_1_name,
        )

        # Cross Attention
        xi = x
        x, cache_c = self.apply(
            inputs=[x, c],
            layer=GatedAttentionUnit_cross_cache,
            units=self.intermediate_size,
            key_size=self.attention_key_size,
            activation=self.hidden_act,
            use_bias=False,
            normalization="softmax_plus",
            attention_dropout=self.attention_dropout_rate,
            kernel_initializer=self.initializer,
            name=cross_attention_name,
            arguments={
                "a_bias": None,
                "p_bias": False,
                "c_cache": c_cache_cross,
            },
        )
        self.cache_outputs.append(cache_c)

        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name="%s-Dropout" % cross_attention_name,
        )
        x = self.apply(inputs=[xi, x], layer=Add, name="%s-Add" % cross_attention_name)
        x = self.apply(
            inputs=x,
            layer=LayerNormalization,
            zero_mean=False,
            scale=False,
            offset=False,
            name="%s-Norm" % cross_attention_name,
        )

        # GAU-2
        xi = x
        x = [x, position_bias]
        arguments = {"a_bias": None, "p_bias": "rotary"}
        if attention_mask is not None:
            arguments["a_bias"] = True
            x.insert(1, attention_mask)
        x, k, v = self.apply(
            inputs=x,
            layer=GatedAttentionUnit_cache,
            units=self.intermediate_size,
            key_size=self.attention_key_size,
            activation=self.hidden_act,
            use_bias=False,
            normalization="softmax_plus",
            attention_dropout=self.attention_dropout_rate,
            kernel_initializer=self.initializer,
            name=self_attention_2_name,
            arguments={
                "a_bias": arguments["a_bias"],
                "p_bias": arguments["p_bias"],
                "k_cache": k_cache_2,
                "v_cache": v_cache_2,
            },
        )
        self.cache_outputs.append(k)
        self.cache_outputs.append(v)
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name="%s-Dropout" % self_attention_2_name,
        )
        x = self.apply(inputs=[xi, x], layer=Add, name="%s-Add" % self_attention_2_name)
        x = self.apply(
            inputs=x,
            layer=LayerNormalization,
            zero_mean=False,
            scale=False,
            offset=False,
            name="%s-Norm" % self_attention_2_name,
        )

        return [c, x]

    def set_outputs(self, outputs):
        """设置output和outputs属性"""
        if not isinstance(outputs, list):
            outputs = [outputs]

        outputs = outputs[:]
        self.outputs = outputs
        self.outputs.extend(self.cache_outputs)

    def build(
        self,
        attention_caches=None,
        layer_norm_cond=None,
        layer_norm_cond_hidden_size=None,
        layer_norm_cond_hidden_act=None,
        additional_input_layers=None,
        **kwargs
    ):
        """模型构建函数
        attention_caches：为Attention的K,V的缓存序列字典，格式为
                         {Attention层名: [K缓存, V缓存]}；
        layer_norm_*系列参数：实现Conditional Layer Normalization时使用，
                            用来实现以“固定长度向量”为条件的条件Bert。
        """
        if self.built:
            return None
        # Input
        inputs = self.get_inputs()
        self.set_inputs(inputs, additional_input_layers)
        # Other
        self.attention_caches = attention_caches or {}
        self.layer_norm_conds = [
            layer_norm_cond,
            layer_norm_cond_hidden_size,
            layer_norm_cond_hidden_act or "linear",
        ]
        # Call
        outputs = self.call(inputs)
        self.inputs.extend(self.cache_inputs)
        self.set_outputs(outputs)
        # Model

        self.model = Model(self.inputs, self.outputs, name=self.name)
        self.built = True
