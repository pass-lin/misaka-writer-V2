import sys
import time
from pathlib import Path

import numpy as np
import pyperclip
import streamlit as st

from backend import (
    device_type,
    generate,
    is_gpu_avaiable,
    load_model,
    model_paths,
    refresh_models,
    tf_version,
)

st.set_page_config(
    page_title="Misaka Writer",
    page_icon="favicon.png",
    layout="wide",
    menu_items={
        "Get Help": "https://github.com/pass-lin/misaka-writer-V2",
        "Report a bug": "https://github.com/pass-lin/misaka-writer-V2/issues",
        "About": "# Misaka Writer V2 \n基于encoder-decoder结构的续写小说模型。",
    },
)

placeholder = """
江染眼底的笑意一闪而过，这才淡淡的道：“我说了，我今天一大早，出现在这里，就是为了来找你。”
“哦？那么你是来找我的吗？”太子殿下眼底微微闪过一抹亮光，一把抓住江染的手腕，“我怎么不记得，你说过什么话？”
""".strip()


def init_state(name, default=None):
    if name not in st.session_state:
        st.session_state[name] = default


init_state("current_model", None)
init_state("outputs", [])
init_state("time_consumed", 0)
init_state("cpu_mode", not is_gpu_avaiable)

with st.sidebar:
    model_path = st.selectbox(
        "选择模型：",
        model_paths,
        help="模型路径在 models 文件夹下。",
        format_func=lambda x: Path(x).name,
    )
    left, right = st.columns(2)
    if left.button("刷新模型列表"):
        refresh_models()
    if right.button("重新加载模型"):
        st.session_state["current_model"] = None

    config_path = st.text_input("配置文件:", value="models/config-misaka.json")
    vocab_path = st.text_input("词表:", value="models/vocab-misaka.txt")

    st.markdown("---")

    max_len = int(
        st.number_input("续写最大长度:", min_value=50, max_value=1500, value=512, step=1)
    )
    nums = int(st.number_input("生成下文数:", min_value=1, max_value=256, value=3, step=1))

    topp = st.number_input(
        "采样阈值(topp):",
        min_value=0.0,
        max_value=1.0,
        value=0.8,
        step=0.01,
        help="采样概率越低，生成的文本内容越丰富，但是越容易出现低质量内容。",
    )
    batch_size = int(
        st.number_input("批大小(batch size):", min_value=1, max_value=256, value=64)
    )
    cpu_mode = st.checkbox(
        "启用针对 CPU 的优化",
        value=st.session_state["cpu_mode"],
        help="基于缓存的优化，可以提升 CPU 的生成速度，对结果有一定影响。",
    )
    if cpu_mode != st.session_state["cpu_mode"]:
        st.session_state["cpu_mode"] = cpu_mode
        st.session_state["current_model"] = None
    if st.session_state["cpu_mode"]:
        repeat_punish = st.number_input(
            "重复惩罚:",
            min_value=0.0,
            max_value=1.0,
            value=0.99,
            step=0.01,
            help="降低重复文本的权重，设为 1.0 表示完全不惩罚。",
        )

    st.markdown("---")

    st.caption(
        f"Tensorflow 版本: {tf_version}<br/>"
        f"设备类型: {device_type}<br/>"
        """<a 
        href="https://github.com/pass-lin/misaka-writer-V2" 
        style="text-decoration: none;color: inherit;">
        Misaka Writer V2
        </a><br/>""",
        unsafe_allow_html=True,
    )

    st.markdown("")

text = st.text_area(
    "输入开头 (建议50~250字):",
    placeholder,
    height=150,
    help="输入过长会显著降低生成结果质量",
)
left, right = st.columns(2)
right.caption(
    f'<div style="text-align: right">当前字数: {len(text)}</div>', unsafe_allow_html=True
)

if model_path:
    model = st.session_state["current_model"]
    with st.spinner("加载模型中..."):
        model = load_model(
            model_path,
            model,
            config_path,
            vocab_path,
            cpu_mode=st.session_state["cpu_mode"],
        )
        st.session_state["current_model"] = model
else:
    st.warning("未找到模型，请将模型放在 models 文件夹下。")
    st.stop()

start_generate = left.button("生成")


class ProgressBar:
    def __init__(self, total):
        self._pbar = st.progress(0.0)
        self._total = total
        self._count = 0
        self._last_time = 0
        self._seconds_per_step = []
        self._eta = st.empty()

    def update(self, count=1):
        self._count += count
        if self._count >= self._total * 0.95:
            self._count = self._total * 0.95
        self._pbar.progress(self._count / self._total)
        with self._eta:
            self._seconds_per_step.append(time.time() - self._last_time)
            if len(self._seconds_per_step) > 15:
                # 神奇 ETA 估算法
                eta = (
                    np.average(self._seconds_per_step)
                    * (self._total - self._count) ** 1.43
                )
                eta = eta * 0.094 + 0.05
                eta = max(eta, 0)
                st.caption(f"预计剩余时间: {int(eta)} 秒")
        self._last_time = time.time()

    def finish(self):
        self._pbar.progress(1.0)
        with self._eta:
            pass

    def __enter__(self):
        self._last_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()


if model and start_generate:
    with ProgressBar(total=max_len * (len(text) // 400 + 1) + 10) as pbar:
        outputs, time_consumed = generate(
            model,
            text,
            nums,
            max_len,
            topp=topp,
            batch_size=batch_size,
            step_callback=lambda nums, n: (
                sys.stderr.write(f"\r[ nums:{nums}   length:{n}]"),
                pbar.update(),
            ),
            cpu_mode=st.session_state["cpu_mode"],
        )
        sys.stderr.write("\n")
        st.session_state["outputs"] = outputs
        st.session_state["time_consumed"] = time_consumed

if st.session_state["outputs"]:
    outputs = st.session_state["outputs"]
    time_consumed = st.session_state["time_consumed"]
    st.success(f"生成完成！耗时：{time_consumed:.2f}s")

    tabs = st.tabs([f"续写{i+1}" for i in range(len(outputs))])
    for i, tab in enumerate(tabs):
        with tab:
            output = outputs[i]
            st.write(output)
            st.button(
                "复制",
                key=i,
                on_click=(lambda output: lambda: pyperclip.copy(output))(output),
            )
