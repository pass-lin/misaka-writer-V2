import collections
import gc
import os
import time
from pathlib import Path
from typing import Any, cast

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import tensorflow as tf

# support tf 2.3+ (use `tf.keras`)
tf_version = tf.version.VERSION
tf_version_tuple = tuple(map(int, tf_version.split(".")))
os.environ["TF_KERAS"] = "1" if tf_version_tuple >= (2, 4, 0) else "0"
from sznlp.my_bert4keras.backend import keras

if tf_version_tuple >= (2, 0, 0):
    tf.compat.v1.disable_eager_execution()
    tf = tf.compat.v1

    def get_session():
        return tf.Session()

else:

    def get_session():
        return keras.backend.get_session()


# 检测 GPU 类型
device_type = tf.test.gpu_device_name()
is_gpu_avaiable = bool(device_type)
device_type = device_type.split(":")[1] if is_gpu_avaiable else "CPU"

# GPU detection
if is_gpu_avaiable:
    gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

Model = collections.namedtuple(
    "Model", ["path", "config", "vocab", "model", "graph", "session"]
)
model_paths = []


def refresh_models():
    global model_paths
    model_paths = [file.as_posix() for file in Path.cwd().rglob("*.h5")]


refresh_models()


def load_model(model_path, model, config_path, vocab_path):
    if (
        model
        and model.path == model_path
        and model.config == config_path
        and model.vocab == vocab_path
    ):
        return model

    from sznlp.my_bert4keras.models import build_transformer_model
    from sznlp.my_bert4keras.tokenizers import Tokenizer

    print(f"Loading model from {model_path}")
    print(f"GPU available: {is_gpu_avaiable}")

    graph = tf.get_default_graph()
    sess = get_session()
    with sess.as_default():
        with graph.as_default():
            tokenizer = Tokenizer(vocab_path, do_lower_case=True)
            if is_gpu_avaiable:
                from sznlp.misaka_models import Misaka
                from sznlp.tools import seq2seq_Generate

                misaka = build_transformer_model(
                    config_path=config_path,
                    model=cast(Any, Misaka),
                    with_lm=True,
                    return_keras_model=False,
                )

                misaka.model.load_weights(model_path, by_name=True)
                encoder = misaka.encoder
                decoder = misaka.decoder
                outputs = [
                    keras.layers.Lambda(lambda x: x[:, -1:])(output)
                    for output in decoder.outputs
                ]
                decoder = keras.models.Model(decoder.inputs, outputs)

                seq2seq = seq2seq_Generate(encoder, decoder, tokenizer)
            else:
                from sznlp.cache_predict import (
                    Misaka_decoder_cache,
                    Misaka_encoder,
                    Seq2SeqGenerate_Cache,
                )

                decoder = build_transformer_model(
                    config_path=config_path,
                    model=cast(Any, Misaka_decoder_cache),
                    with_lm=True,
                    return_keras_model=True,
                )

                encoder = build_transformer_model(
                    config_path=config_path,
                    model=cast(Any, Misaka_encoder),
                    with_lm=True,
                    return_keras_model=True,
                )

                decoder.load_weights(model_path, by_name=True)
                encoder.load_weights(model_path, by_name=True)

                seq2seq = Seq2SeqGenerate_Cache(
                    encoder, decoder, tokenizer, skip_token="氼"
                )

    model = Model(model_path, config_path, vocab_path, seq2seq, graph, sess)
    print("Model loaded. ")
    return model


def generate(
    model,
    text,
    nums,
    max_len,
    topp=0.8,
    batch_size=32,
    repeat_punish=0.99,
    step_callback=None,
):
    if not model:
        return ["模型加载中，请稍候..."], 0.0

    if (2, 0, 0) <= tf_version_tuple < (2, 4, 0):
        keras.backend.tensorflow_backend._SYMBOLIC_SCOPE.value = True  # type: ignore

    start_time = time.time()

    with model.session.as_default():
        with model.graph.as_default():
            if is_gpu_avaiable:
                result = model.model.writer(
                    [text.replace("\n", "氼")],  # 文本数据就是上面的data
                    nums=nums,  # 一个开头要生成几个文本
                    k=topp,  # 搜索窗口
                    batch_size=batch_size,
                    max_len=max_len,  # 最大长度
                    iter_data_num=400,  # 一次处理多少个开头
                    mode="topp",  # 别动的句子的次数，越大就越慢同时重复句子越少)
                    iter_max_num=0,
                    step_callback=step_callback,
                )  # 检查重复解码
            else:
                result = model.model.writer(
                    [text.replace("\n", "氼")],  # 文本数据就是上面的data
                    nums=nums,  # 输入要生成几个文本
                    k=topp,
                    batch_size=batch_size,
                    max_len=max_len,
                    repeat_punish=repeat_punish,
                    step_callback=step_callback,
                )  # 检查重复解码
    generated = ["\n\n".join(result[i].split("氼")) for i in range(nums)]
    time_consumed = time.time() - start_time
    return generated, time_consumed


def cleanup_memory():
    keras.backend.clear_session()
    gc.collect()
