# -*- coding: utf-8 -*-
"""
Created on Wed May 25 13:47:50 2022

@author: Administrator
"""
#import os
#os.environ['TF_KERAS'] = '1'
from my_bert4keras.models import *
class GatedAttentionUnit_cross(Layer):
    """门控注意力单元
    链接：https://arxiv.org/abs/2202.10447
    介绍：https://kexue.fm/archives/8934
    在苏神基础上支持cross-attention
    """
    def __init__(
        self,
        units,
        key_size,
        activation='swish',
        use_bias=True,
        normalization='squared_relu',
        attention_scale=True,
        attention_dropout=None,
        kernel_initializer='glorot_uniform',
        **kwargs
    ):
        super(GatedAttentionUnit_cross, self).__init__(**kwargs)
        self.units = units
        self.key_size = key_size
        self.activation = activation
        self.use_bias = use_bias
        self.normalization = normalization
        self.attention_scale = attention_scale
        self.attention_dropout = attention_dropout
        self.kernel_initializer = initializers.get(kernel_initializer)
    def initializer(self, shape, dtype=None, order=3, gain=1.0):
        if shape[0] > 10000 or shape[0] < 10:
            hidden_size = shape[1]
        else:
            hidden_size = shape[0]
        gain *= (self.num_hidden_layers*5/2)**(-1. / order)
        stddev = 1.13684723 / hidden_size**0.5 * gain
        return K.truncated_normal(shape, stddev=stddev)
    @integerize_shape
    def build(self, input_shape):
        super(GatedAttentionUnit_cross, self).build(input_shape)
        hidden_size = input_shape[-1]
        if isinstance(hidden_size, (list, tuple)):
            hidden_size = input_shape[0][-1]
        self.kv_dense = Dense(
            units=self.units + self.key_size,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )
        self.uq_dense = Dense(
            units=self.units + self.key_size,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )
        self.o_dense = Dense(
            units=hidden_size,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )


    @recompute_grad
    def call(self, inputs, mask=None, a_bias=False, p_bias=None):
        if not isinstance(inputs, list):
            inputs, mask = [inputs], [mask]
        x,c= inputs[:]
        n=1
        mask = None if mask is None else mask[1]
        if a_bias:
            a_bias = inputs[n]
            n += 1
        # 投影变换
        x = self.uq_dense(x)
        u,q = tf.split(x, [self.units, self.key_size], axis=-1)
        
        c=self.kv_dense(c)
        v,k = tf.split(c, [self.units, self.key_size], axis=-1)
        # 加入RoPE
        if p_bias == 'rotary':
            q, k = apply_rotary_position_embeddings(inputs[n], q, k)
        # Attention
        a = tf.einsum('bmd,bnd->bmn', q, k)
        if self.attention_scale:
            a = a / self.key_size**0.5
        if a_bias is not None:
            a = a + a_bias
        a = sequence_masking(a, mask, '-inf', -1)
        A = attention_normalize(a, -1, self.normalization)
        if self.attention_dropout:
            A = Dropout(self.attention_dropout)(A)
        # 计算输出
        o = self.o_dense(u * tf.einsum('bmn,bnd->bmd', A, v))
        return o

    def compute_mask(self, inputs, mask=None):
        return mask
    def compute_output_shape(self, input_shape):
        if isinstance(input_shape[0], (list, tuple)):
            return input_shape[0]
        else:
            return input_shape

    def get_config(self):
        config = {
            'units': self.units,
            'key_size': self.key_size,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'normalization': self.normalization,
            'attention_scale': self.attention_scale,
            'attention_dropout': self.attention_dropout,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
        }
        base_config = super(GatedAttentionUnit_cross, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
class Misaka_Base(RoFormerV2):
    def initializer(self, shape, dtype=None, order=3, gain=1.0):
        return super(Misaka_Base, self).initializer(shape, dtype, order, gain)
    def variable_mapping(self):
        pass
class Misaka_encoder(Misaka_Base):
    """基于GAU-α的encoder
    链接：https://kexue.fm/archives/9052
    """
    def get_inputs(self):
        """Misaka的Encoder的输入只有token_ids
        """
        x_in = self.apply(
            layer=Input,
            shape=(self.sequence_length,),
            name='Encoder-Input-Token'
        )
        return x_in
    def apply_embeddings(self, inputs):
        """
        Misaka embeding只有word embeding
        """
        x=inputs

        x = self.apply(
            inputs=x,
            layer=Embedding,
            input_dim=self.vocab_size,
            output_dim=self.embedding_size,
            embeddings_initializer=self.initializer,
            mask_zero=True,
            name='Embedding-Token'
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='Embedding-Dropout'
        )
        if self.embedding_size != self.hidden_size:
            x = self.apply(
                inputs=x,
                layer=Dense,
                units=self.hidden_size,
                use_bias=False,
                kernel_initializer=self.initializer,
                name='Embedding-Mapping'
            )

        return x
    def apply_main_layers(self, inputs, index):
        """Misaka-encoder 的主体是基于Gated Attention Unit的模块
        顺序：GAU  --> Add --> LN
        """
        x = inputs

        attention_name = 'Misaka-Encoder-%d-GatedAttentionUnit' % index
        attention_mask = self.compute_attention_bias(index)
        position_bias = self.compute_position_bias(x)
        
        # Self Attention
        xi = x
        x = [x, position_bias]
        arguments = {'a_bias': None, 'p_bias': 'rotary'}
        if attention_mask is not None:
            arguments['a_bias'] = True
            x.insert(1, attention_mask)
        x = self.apply(
            inputs=x,
            layer=GatedAttentionUnit,
            arguments=arguments,
            units=self.intermediate_size,
            key_size=self.attention_key_size,
            activation=self.hidden_act,
            use_bias=False,
            normalization='softmax_plus',
            attention_dropout=self.attention_dropout_rate,
            kernel_initializer=self.initializer,
            name=attention_name
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % attention_name
        )
        x = self.apply(
            inputs=[xi, x], layer=Add, name='%s-Add' % attention_name
        )
        x = self.apply(
            inputs=x,
            layer=LayerNormalization,
            zero_mean=False,
            scale=False,
            offset=False,
            name='%s-Norm' % attention_name
        )

        return x
    def apply_final_layers(self, inputs):
        """剩余部分
        """
        x = inputs
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='Encoder-Output-Dropout'
        )
        return x
class Misaka_decoder(LM_Mask,Misaka_Base):
    """Misaka模型（Decoder）
    """
    def __init__(self, with_lm=True, **kwargs):
        super(Misaka_decoder, self).__init__(**kwargs)
        self.with_lm = with_lm
        self.num_hidden_layers=self.num_hidden_layers//2
    def initializer(self, shape, dtype=None, order=3, gain=1.0):
        if shape[0] > 10000 or shape[0] < 10:
            hidden_size = shape[1]
        else:
            hidden_size = shape[0]
        gain *= (self.num_hidden_layers*5)**(-1. / order)
        stddev = 1.13684723 / hidden_size**0.5 * gain
        return K.truncated_normal(shape, stddev=stddev)
    def apply_embeddings(self, inputs):
        c, x = inputs

        x = self.apply(
            inputs=x,
            layer=Embedding,
            input_dim=self.vocab_size,
            output_dim=self.embedding_size,
            embeddings_initializer=self.initializer,
            mask_zero=True,
            name='Embedding-Token'
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='Decoder-Embedding-Dropout'
        )
        if self.embedding_size != self.hidden_size:
            x = self.apply(
                inputs=x,
                layer=Dense,
                units=self.hidden_size,
                kernel_initializer=self.initializer,
                name='Decoder-Embedding-Mapping'
            )

        return [c, x]
    def get_inputs(self):
        """Misaka的Decoder的输入为context序列和token_ids
        """
        c_in = self.apply(
            layer=Input,
            shape=(self.sequence_length, self.hidden_size),
            name='Input-Context'
        )
        x_in = self.apply(
            layer=Input,
            shape=(self.sequence_length,),
            name='Decoder-Input-Token'
        )
        return [c_in, x_in]
    def apply_main_layers(self, inputs, index):
        """Misaka-encoder 的主体是基于Gated Attention Unit的模块
        顺序：LN --> GAU1 --> Add --> LN --> cross-attention  --> Add -->  LN --> GAU  --> Add
        其中cross-attention我使用的是自己改的GAU
        """
        c, x  = inputs[:]
        
        self_attention_1_name='Misaka-Dncoder-%d-GatedAttentionUnit-1' % index
        cross_attention_name = 'Misaka-Dncoder-%d-GatedAttentionUnit-cross' % index
        self_attention_2_name='Misaka-Dncoder-%d-GatedAttentionUnit-2' % index
        attention_mask = self.compute_attention_bias(index)
        position_bias = self.compute_position_bias(x)

        # GAU-1
        xi = x
        x = [x, position_bias]
        arguments = {'a_bias': None, 'p_bias': 'rotary'}
        if attention_mask is not None:
            arguments['a_bias'] = True
            x.insert(1, attention_mask)
        
        x = self.apply(
            inputs=x,
            layer=GatedAttentionUnit,
            arguments=arguments,
            units=self.intermediate_size,
            key_size=self.attention_key_size,
            activation=self.hidden_act,
            use_bias=False,
            normalization='softmax_plus',
            attention_dropout=self.attention_dropout_rate,
            kernel_initializer=self.initializer,
            name=self_attention_1_name
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % self_attention_1_name
        )
        x = self.apply(
            inputs=[xi, x], layer=Add, name='%s-Add' % self_attention_1_name
        )
        x = self.apply(
            inputs=x,
            layer=LayerNormalization,
            zero_mean=False,
            scale=False,
            offset=False,
            name='%s-Norm' % self_attention_1_name
        )
        
        # Cross Attention
        xi=x
        x = self.apply(
            inputs=[x,c],
            layer=GatedAttentionUnit_cross,
            units=self.intermediate_size,
            key_size=self.attention_key_size,
            activation=self.hidden_act,
            use_bias=False,
            normalization='softmax_plus',
            attention_dropout=self.attention_dropout_rate,
            kernel_initializer=self.initializer,
            name=cross_attention_name
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % cross_attention_name
        )
        x = self.apply(
            inputs=[xi, x], layer=Add, name='%s-Add' % cross_attention_name
        )
        x = self.apply(
            inputs=x,
            layer=LayerNormalization,
            zero_mean=False,
            scale=False,
            offset=False,
            name='%s-Norm' % cross_attention_name
        )
        
        # GAU-2
        xi = x
        x = [x, position_bias]
        arguments = {'a_bias': None, 'p_bias': 'rotary'}
        if attention_mask is not None:
            arguments['a_bias'] = True
            x.insert(1, attention_mask)
        x = self.apply(
            inputs=x,
            layer=GatedAttentionUnit,
            arguments=arguments,
            units=self.intermediate_size,
            key_size=self.attention_key_size,
            activation=self.hidden_act,
            use_bias=False,
            normalization='softmax_plus',
            attention_dropout=self.attention_dropout_rate,
            kernel_initializer=self.initializer,
            name=self_attention_2_name
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % self_attention_2_name
        )
        x = self.apply(
            inputs=[xi, x], layer=Add, name='%s-Add' % self_attention_2_name
        )
        x = self.apply(
            inputs=x,
            layer=LayerNormalization,
            zero_mean=False,
            scale=False,
            offset=False,
            name='%s-Norm' % self_attention_2_name
        )

        return [c, x]

    def apply_final_layers(self, inputs):
        """剩余部分
        """
        c,x = inputs

        if self.with_lm:
            # 预测token概率部分
            if self.embedding_size != self.hidden_size:
                x = self.apply(
                    inputs=x,
                    layer=Dense,
                    units=self.embedding_size,
                    use_bias=False,
                    kernel_initializer=self.initializer,
                    name='Output-Mapping'
                )
            x = self.apply(
                inputs=x,
                layer=Dropout,
                rate=self.dropout_rate,
                name='Output-Output-Dropout'
            )
            Output_activation = 'softmax' if self.with_lm is True else self.with_lm
            
            x = self.apply(
                    inputs=x,
                    layer=Dense,
                    units=self.vocab_size,
                    activation= Output_activation,
                    use_bias=False,
                    kernel_initializer=self.initializer,
                    name='Decoder-Output-LM'
                )
        return x
    def compute_attention_bias(self, inputs=None):
        """修改LM Mask的序列长度（从 self.inputs[0] 改为 self.inputs[1] ）
        """
        old_inputs = self.inputs[:]
        self.inputs = [old_inputs[1]]
        mask = super(Misaka_decoder, self).compute_attention_bias(inputs)
        self.inputs = old_inputs
        return mask
class Misaka(Misaka_Base):
    """Misaka模型（Encoder-Decoder）
    """
    def __init__(self, **kwargs):
        super(Misaka, self).__init__(**kwargs)
        kwargs['layers'] = self.layers
        e_name, d_name = 'Misaka_encoder', 'Misaka_decoder'
        if 'name' in kwargs:
            e_name = '%s_%s' % (kwargs['name'], e_name)
            d_name = '%s_%s' % (kwargs['name'], d_name)
            del kwargs['name']  # 防止重复传参
        self._encoder = Misaka_encoder(name=e_name, **kwargs)
        self._decoder = Misaka_decoder(name=d_name, **kwargs)
    
    def build(self, **kwargs):
        """同时构建Encoder和Decoder
        """
        self._encoder.build(**kwargs)
        self._decoder.build(**kwargs)
        self._decoder.position_bias = None  # 下面call时将重新初始化
        self.encoder = self._encoder.model
        self.decoder = self._decoder.model
        self.inputs = self.encoder.inputs + self.decoder.inputs[1:]
        self.outputs = self._decoder.call(
            self.encoder.outputs + self.decoder.inputs[1:]
        )
        self.model = Model(self.inputs, self.outputs)
class Misaka_decoder_NAT(Misaka_decoder):
    def compute_attention_bias(self, inputs=None):
        return None
class Misaka_NAT(Misaka_Base):
    def __init__(self, **kwargs):
        super(Misaka, self).__init__(**kwargs)
        kwargs['layers'] = self.layers
        e_name, d_name = 'Misaka_encoder', 'Misaka_decoder'
        if 'name' in kwargs:
            e_name = '%s_%s' % (kwargs['name'], e_name)
            d_name = '%s_%s' % (kwargs['name'], d_name)
            del kwargs['name']  # 防止重复传参
        self._encoder = Misaka_encoder(name=e_name, **kwargs)
        self._decoder = Misaka_decoder_NAT(name=d_name, **kwargs)
    
    def build(self, **kwargs):
        """同时构建Encoder和Decoder
        """
        self._encoder.build(**kwargs)
        self._decoder.build(**kwargs)
        self._decoder.position_bias = None  # 下面call时将重新初始化
        self.encoder = self._encoder.model
        self.decoder = self._decoder.model
        self.inputs = self.encoder.inputs + self.decoder.inputs[1:]
        self.outputs = self._decoder.call(
            self.encoder.outputs + self.decoder.inputs[1:]
        )
        self.model = Model(self.inputs, self.outputs)