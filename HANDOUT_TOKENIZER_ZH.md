# CS336 Assignment 1 Handout 中文导读：Tokenizer / BPE

来源：`cs336_assignment1_basics.pdf`，Section 2 Byte-Pair Encoding (BPE) Tokenizer。

说明：这不是 PDF 的逐字完整翻译，而是面向完成作业的中文导读。它保留原章节结构、核心概念、接口要求和实现注意点，避免把 handout 大段复刻成替代原文。

## 2. Byte-Pair Encoding Tokenizer 总览

作业第一部分要求训练并实现一个 byte-level BPE tokenizer。

核心思想：

```text
任意 Unicode 字符串
    -> 表示成 UTF-8 bytes
    -> 在 byte 序列上训练 BPE tokenizer
    -> 之后用 tokenizer 把文本编码成整数 token ids
```

语言模型最终需要的是整数序列，而不是字符串。tokenizer 负责把文本转成 token id，也负责把 token id 解码回文本。

## 2.1 Unicode Standard

Unicode 是一个字符标准。它把字符映射到整数 code point。

例子：

- 字符 `"s"` 的 code point 是 115，也常写作 `U+0073`。
- 字符 `"牛"` 的 code point 是 29275。

Python 中：

```python
ord("牛")
```

表示从字符得到整数 code point。

```python
chr(29275)
```

表示从 code point 得到字符。

学习重点：

- Unicode 字符和整数 code point 是一层映射。
- 但 tokenizer 通常不直接在 Unicode code point 上训练，因为 code point 空间大且稀疏。

## 2.2 Unicode Encodings

Unicode 标准定义字符到 code point，但实际存储和传输通常需要编码成 bytes。常见编码包括 UTF-8、UTF-16、UTF-32，其中 UTF-8 是互联网上最常用的编码。

Python 中：

```python
text.encode("utf-8")
```

把字符串转成 bytes。

```python
data.decode("utf-8")
```

把 UTF-8 bytes 转回字符串。

重要直觉：

```text
一个 Unicode 字符不一定等于一个 byte。
```

英文字符通常是 1 byte，但中文、日文、emoji 往往是多个 bytes。

byte-level tokenizer 的优势：

```text
所有普通文本都可以表示成 0..255 的 byte 序列。
```

因此，只要 vocab 初始包含 256 个 byte token，就不会有普通文本的 out-of-vocabulary 问题。

decode 注意点：

不要把每个 byte 单独 decode 再拼字符串。某些 Unicode 字符由多个 bytes 组成，单独 decode 会失败或出错。正确方向是：

```text
先拼完整 bytes
再统一 decode("utf-8")
```

## 2.3 Subword Tokenization

纯 byte-level tokenization 虽然不会 OOV，但序列会很长。长序列会让训练更慢，也让模型更难学习长距离依赖。

subword tokenization 是 word-level 和 byte-level 之间的折中：

```text
byte-level:
    vocab 小，序列长，不会 OOV

word-level:
    vocab 大，序列短，但容易 OOV

subword:
    vocab 适中，序列较短，OOV 问题小
```

BPE 的作用是把训练语料里频繁出现的相邻 byte/token pair 合并成更长 token。

例子：

```text
b"t" + b"h" -> b"th"
b"th" + b"e" -> b"the"
```

如果 `b"the"` 很常见，把它加入 vocab 可以把三个 byte token 压成一个 token。

## 2.4 BPE Tokenizer Training

BPE training 的目标：

```text
输入训练语料
    -> 输出 vocab 和 merges
```

它不同于 tokenizer inference。training 是学规则；inference 是用规则。

### 2.4.1 Vocabulary initialization

因为是 byte-level BPE，初始 vocab 包含所有 256 个 byte。

如果有 special tokens，也要加入 vocab，让它们拥有固定 token id。

概念上：

```text
初始 vocab = special tokens + 256 byte tokens
```

### 2.4.2 Pre-tokenization

如果直接在整篇语料上统计 byte pair，每次 merge 都要重新扫描全文，很慢，而且容易学到跨越奇怪边界的 token。

pretokenization 是粗粒度切分：

```text
corpus
    -> pretokens
    -> 每个 pretoken 转 UTF-8 bytes
    -> 在 pretoken 内部统计 pair
```

好处：

- 不跨 pretoken 边界 merge。
- 可以统计 pretoken 频率，减少重复工作。
- 更贴近现代 tokenizer 的行为。

作业采用 GPT-2 风格 regex pretokenizer：

```python
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

需要使用第三方 `regex` 包，而不是内置 `re`，因为它支持 `\p{L}` 和 `\p{N}` 这类 Unicode property。

示例直觉：

```text
"some text that i'll pre-tokenize"
    -> "some", " text", " that", " i", "'ll", " pre", "-", "tokenize"
```

注意：很多 pretoken 会带前导空格，比如 `" text"`。

### 2.4.3 Compute BPE merges

训练时反复做：

```text
统计所有 pretoken 内部的相邻 pair 频率
选择最高频 pair
把这个 pair 合并成新 token
把新 token 加入 vocab
把这次 merge 记录到 merges
```

直到达到目标 vocab size。

重要规则：

- 不考虑跨 pretoken 边界的 pair。
- 如果多个 pair 频率并列，按作业要求用确定性 tie-break。
- merges 的顺序就是之后 encode 时的优先级。

### 2.4.4 Special tokens in training

special token 例如 `<|endoftext|>` 通常表示文档边界或控制信息。

训练时：

- special token 要加入 vocab。
- special token 是硬边界，防止跨文档 merge。
- special token 不参与普通 BPE pair frequency 统计。

也就是说：

```text
[Doc 1]<|endoftext|>[Doc 2]
```

应该分开处理 `[Doc 1]` 和 `[Doc 2]`，不能让 merge 跨过 `<|endoftext|>`。

## 2.5 Experimenting With BPE Tokenizer Training

这一节要求你之后在 TinyStories 和 OpenWebText 上训练 tokenizer，并记录时间、内存、最长 token、compression ratio 等。

和当前实现最相关的提示：

- pretokenization 可能是训练瓶颈。
- 可以用 multiprocessing 并行 pretokenization。
- chunk boundary 应该落在 special token 边界上。
- 仓库里的 `cs336_basics/pretokenization_example.py` 给了 chunk boundary 的参考代码。

## 2.6 BPE Tokenizer: Encoding and Decoding

这一节对应当前正在实现的 Tokenizer class。

它的目标：

```text
给定 vocab 和 merges
实现 encode / decode
```

注意这里是使用已有规则，不是训练规则。

### 2.6.1 Encoding text

encode 的流程：

```text
输入 text
    -> pretokenize
    -> 每个 pretoken 表示成 UTF-8 bytes
    -> 在每个 pretoken 内部应用 merges
    -> 最终 bytes token 查 vocab 得到 token id
    -> 输出 list[int]
```

边界规则：

- 每个 pretoken 独立处理。
- 不跨 pretoken 边界 merge。
- special token 要单独处理，不能被普通 pretokenization 和 BPE 拆开。

作业 handout 的例子说明：

```text
"the cat ate"
    -> "the", " cat", " ate"
```

然后每个 pretoken 内部按 merges 合并，最后得到 token id 序列。

### 2.6.2 Memory considerations

大文件不能总是一次性读入内存。

因此 tokenizer 还需要：

```python
encode_iterable(iterable)
```

它应该懒惰地产生 token ids，而不是一次性返回全部 ids。

难点是：随便按 chunk 分开 encode 可能改变跨 chunk 边界的 tokenization，所以要小心边界处理。

### 2.6.3 Decoding text

decode 的流程：

```text
ids
    -> vocab[id] 得到 bytes
    -> 拼接所有 bytes
    -> decode 成 Unicode 字符串
```

如果 ids 对应的 bytes 不是合法 UTF-8，作业要求用 replacement character 处理 malformed bytes。Python 中可以通过 bytes decode 的错误处理参数完成。

## 2.6 Tokenizer Interface

handout 推荐 Tokenizer class 提供这些方法：

```text
__init__(vocab, merges, special_tokens=None)
from_files(vocab_filepath, merges_filepath, special_tokens=None)
encode(text: str) -> list[int]
encode_iterable(iterable: Iterable[str]) -> Iterator[int]
decode(ids: list[int]) -> str
```

当前测试通过 adapter 调用：

```text
tests.adapters.get_tokenizer(...)
```

所以 adapter 负责返回你实现的 Tokenizer 对象。

## 当前实现顺序

建议按这个顺序写：

```text
1. Tokenizer 初始化：保存 vocab，构造 reverse_vocab，构造 merge ranks。
2. Decode：ids -> bytes -> str。
3. Pretokenize：用 PAT 切普通文本。
4. 单个 pretoken 的 BPE merge。
5. 普通文本 encode。
6. Special token 分割和整体映射。
7. encode_iterable。
8. BPE training。
```

先把 tokenizer inference 完成，再做 `run_train_bpe`。
