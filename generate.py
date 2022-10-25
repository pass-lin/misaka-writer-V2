# -*- coding: utf-8 -*-
import sys
import textwrap

from backend import generate, load_model

try:
    from alive_progress import alive_bar
except ImportError:
    alive_bar = None

if __name__ == "__main__":
    model_path = "models/古言.h5"  # 模型路径
    nums = 1  # 开头生成下文的数量
    max_len = 512  # 最大长度
    topp = 0.8  # 采样概率
    batch_size = 32  # 批大小
    # 开头，建议开头字数在50字到200字之间
    text = """
    白月耸了耸肩膀，无语。黎傲然不再说话，继续闭上眼睛养起神来。
    马车慢慢的驶出了城，在城外宽阔的大道上前行着。
    “咦？”凌言看着窗外越来越僻静的小路，觉察出了不对劲。
    “怎么了？”白月不解的看着凌言。
    “似乎方向不对啊。”凌言将头探出窗外，大声冲车夫道，“师傅，你是不是走错路了？”
    车夫却丝毫不理会凌言的话，反而扬起鞭子抽了马一鞭子，将马车赶的更快了。
    """
    output = "out.txt"  # 输出文件名

    # 加载模型
    model = load_model(
        model_path, None, "models/config-misaka.json", "models/vocab-misaka.txt"
    )
    # 生成
    text = textwrap.dedent(text)

    if alive_bar is not None:
        elasped_total = max_len * (len(text) // 400 + 1) + 10
        with alive_bar(total=elasped_total, title="Generating", dual_line=True) as bar:
            outputs, time_consumed = generate(
                model,
                text,
                nums,
                max_len,
                topp=topp,
                batch_size=batch_size,
                step_callback=lambda nums, _: (
                    bar.text(f"Remaining nums: {nums}"),
                    bar(),
                ),
            )
            while bar.current() < elasped_total:
                bar()

    else:
        outputs, time_consumed = generate(
            model,
            text,
            nums,
            max_len,
            topp=topp,
            batch_size=batch_size,
            step_callback=lambda nums, n: sys.stderr.write(
                f"\r[ nums:{nums}   length:{n}]"
            ),
        )

    sys.stderr.write(f"Finished in {time_consumed:.2f}s.\n")

    # 输出
    with open(output, "w", encoding="utf-8") as f:
        for _ in range(nums):
            f.write(textwrap.indent(text, "\t") + "\n")
            for output in outputs:
                f.write(textwrap.indent(output, "\t") + "\n")
            f.write("\n" + "*" * 80 + "\n")
