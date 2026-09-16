# CS336 Assignment 1 Learning Plan

更新时间：2026-09-13

这份文件用于记录 Assignment 1 的学习路线、当前进度、每个模块的讲解摘要和复习要点。后续学习时，每完成一个模块或进入新阶段，都在这里更新进度。

原则：

- 先理解概念和测试要求，再写代码。
- 每次只推进一个模块，避免同时调试太多问题。
- 代码由你自己写；这里记录思路、检查点和测试路径，不保存完整作业答案。
- 遇到失败测试时，先读测试，再定位 adapter，再检查输入输出 shape 和边界条件。

## Progress

| 阶段 | 模块 | 状态 | 相关测试 |
| --- | --- | --- | --- |
| 1 | 项目机制 | Done | `uv run pytest -q` |
| 2 | Tokenizer / BPE | Done | `uv run pytest tests/test_tokenizer.py -q`; `uv run pytest tests/test_train_bpe.py -q` |
| 3 | Data batching | Done | `uv run pytest tests/test_data.py -q` |
| 4 | 基础 NN 组件 | Done | `uv run pytest tests/test_model.py -q`; `uv run pytest tests/test_nn_utils.py -q` |
| 5 | Attention / RoPE | In progress | `uv run pytest tests/test_model.py::test_scaled_dot_product_attention -q`; `uv run pytest tests/test_model.py::test_rope -q` |
| 6 | Transformer block / Transformer LM | Not started | `uv run pytest tests/test_model.py::test_transformer_block -q`; `uv run pytest tests/test_model.py::test_transformer_lm -q` |
| 7 | Optimizer / scheduler / checkpoint | Not started | `uv run pytest tests/test_optimizer.py -q`; `uv run pytest tests/test_serialization.py -q` |
| 8 | 整体验证和提交 | Not started | `uv run pytest -q`; `./make_submission.sh` |

## One-Week Completion Plan

目标：一周内完成 Assignment 1，但每个模块都先学清楚再写代码。

节奏安排：

| 天数 | 目标 | 输出 |
| --- | --- | --- |
| Day 1 | Tokenizer/BPE 概念课 + tokenizer inference | 通过 `tests/test_tokenizer.py` 的 encode/decode 主体测试 |
| Day 2 | BPE training | 通过 `tests/test_train_bpe.py`，并理解速度瓶颈 |
| Day 3 | Data batching + 基础 NN primitives | 通过 `tests/test_data.py`、`tests/test_nn_utils.py` 和基础 model tests |
| Day 4 | Attention + RoPE | 通过 SDPA、MHA、RoPE 相关测试 |
| Day 5 | Transformer block + Transformer LM | 通过 full model forward 测试 |
| Day 6 | Optimizer、scheduler、checkpoint | 通过 optimizer 和 serialization 测试 |
| Day 7 | 全量测试、清理、提交 | `uv run pytest -q` 通过，生成提交包 |

后续固定流程：

```text
先讲模块知识
    -> 再读测试约束
    -> 你写代码
    -> 我帮你看失败和边界情况
    -> 更新本计划进度
```

## Reference Materials

使用原则：

- 优先看官方课程站、官方 handout、官方 lecture materials。
- 可以看概念型笔记和公开视频来补理解。
- 不看第三方完整作业实现或直接可复制的解法代码。课程政策明确说 handout 是自包含的，现成实现不应作为作业实现参考。

### Official

- CS336 current course site: https://cs336.stanford.edu/
- CS336 Spring 2025 archive: https://cs336.stanford.edu/spring2025/
- Assignment 1 official repo: https://github.com/stanford-cs336/assignment1-basics
- Official lecture materials repo: https://github.com/stanford-cs336/lectures
- Lecture 1 trace/preview: https://cs336.stanford.edu/lectures/?trace=lecture_01
- Assignment 1 local handout: `cs336_assignment1_basics.pdf`

### Useful For Assignment 1

- Lecture 1: Overview, tokenization.
- Lecture 2: PyTorch, einops, resource accounting.
- Lecture 3: Architectures and hyperparameters.
- Lecture 10: Inference, useful later for generation/sampling intuition.

### External Background

- BPE subword paper: https://aclanthology.org/P16-1162/
- OpenAI GPT-2 tokenizer source, for conceptual comparison only: https://github.com/openai/gpt-2/blob/master/src/encoder.py
- OpenAI tiktoken educational code, for conceptual comparison only: https://github.com/openai/tiktoken/blob/main/tiktoken/_educational.py
- PyTorch docs: https://pytorch.org/docs/stable/index.html
- einops docs: https://einops.rocks/

### Notes And Summaries

- Qihong Ruan CS336 notes: https://qihongruan.github.io/cs336/
- Open Course Notes CS336 2026 Chinese notes: https://yulong-ge.github.io/open-course-notes/courses/stanford-cs336-2026/
- Lecture Atlas overview: https://chadsmith.dev/lecture-atlas/
- Class Central playlist index: https://www.classcentral.com/course/youtube-stanford-cs336-language-modeling-from-scratch-i-2025-512656

### Avoid For Implementation

- Public forks or blog posts that include completed Assignment 1 code.
- Any repository whose main value is passing CS336 tests.
- Code snippets that directly implement tokenizer, Transformer, AdamW, RoPE, or training loop for this exact assignment.

## Test Coverage Audit

本计划已按 `tests/adapters.py` 和 `tests/test_*.py` 做过覆盖检查。每个测试入口对应的学习模块如下：

| Adapter / 测试入口 | 对应模块 | 备注 |
| --- | --- | --- |
| `get_tokenizer` | Module 2 | tokenizer inference：`encode`、`decode`、`encode_iterable` |
| `run_train_bpe` | Module 2 | BPE training；放在 tokenizer inference 之后 |
| `run_get_batch` | Module 3 | 语言模型训练样本采样 |
| `run_linear` | Module 4 | 基础 NN 组件 |
| `run_embedding` | Module 4 | 基础 NN 组件 |
| `run_silu` | Module 4 | 基础 NN 组件 |
| `run_softmax` | Module 4 | 基础 NN 组件；包含数值稳定性 |
| `run_cross_entropy` | Module 4 | 基础 NN 组件；包含数值稳定性 |
| `run_rmsnorm` | Module 4 | 基础 NN 组件 |
| `run_swiglu` | Module 4 | FFN 组件 |
| `run_scaled_dot_product_attention` | Module 5 | 同时覆盖普通 shape 和 4D batch/head shape |
| `run_multihead_self_attention` | Module 5 | attention 组合 |
| `run_rope` | Module 5 | 位置编码 |
| `run_multihead_self_attention_with_rope` | Module 5 | attention + RoPE |
| `run_transformer_block` | Module 6 | 单个 pre-norm block |
| `run_transformer_lm` | Module 6 | 完整 LM forward；包含短于 context length 的输入 |
| `run_gradient_clipping` | Module 7 | 梯度处理 |
| `get_adamw_cls` | Module 7 | 优化器 |
| `run_get_lr_cosine_schedule` | Module 7 | 学习率 schedule |
| `run_save_checkpoint` | Module 7 | 序列化 |
| `run_load_checkpoint` | Module 7 | 反序列化 |

需要特别提前记住的测试约束：

- Tokenizer: `encode_iterable` 有内存测试，设计时不能依赖一次性读完整大文件。
- Tokenizer: 多个测试会与 `tiktoken.get_encoding("gpt2")` 对齐。
- Tokenizer: special token 有重叠和换行边界测试。
- BPE training: 有速度测试，不能只用非常低效的玩具算法。
- Data batching: 起点采样必须覆盖合法范围，且 `y` 必须是 `x` 右移一位。
- Softmax / cross entropy: 测试包含大 logits，需要数值稳定。
- Attention: SDPA 测试包含 4D 输入，不能只写死二维或三维情况。
- Transformer LM: 测试包含短于 `context_length` 的输入。
- Checkpoint: 需要恢复 model state、optimizer state 和 iteration。

状态说明：

- `Not started`: 还没开始。
- `In progress`: 正在学习或实现。
- `Blocked`: 有问题需要先解决。
- `Done`: 概念讲解、代码实现、相关测试都完成。

## Module 1: 项目机制

状态：Done

目标：理解这个作业仓库如何组织，以及测试如何调用你的实现。

核心结构：

```text
tests/test_*.py
        ↓
tests/adapters.py
        ↓
你在 cs336_basics/ 中写的实现
```

关键文件：

- `README.md`: 项目说明、环境设置、测试命令、数据下载方式。
- `pyproject.toml`: Python 项目配置，声明项目名、Python 版本、依赖、pytest 和 ruff 配置。
- `uv.lock`: 锁定依赖版本，保证环境可复现。
- `tests/adapters.py`: 测试调用你的实现的接口层，是整份作业最重要的 TODO 清单。
- `tests/test_*.py`: 自动测试，按模块验证 tokenizer、model、optimizer、serialization 等功能。
- `cs336_basics/`: 建议存放你自己实现的代码。
- `cs336_assignment1_basics.pdf`: 作业 handout，包含正式要求。
- `make_submission.sh`: 提交前测试并打包 zip 的脚本。

已经讲过的概念：

- `uv run pytest -q` 表示在 uv 管理的项目环境里运行 pytest，并使用简洁输出。
- `pytest` 会按规则发现测试文件，通常包括 `test_*.py` 和 `*_test.py`。
- 测试文件不一定必须放在 `tests/` 下，但项目通常这样组织更清楚。
- `pyproject.toml` 是 Python 项目的统一配置中心，`uv`、`pytest`、`ruff` 都可以从里面读取配置。
- `pytest` 是测试框架；`ruff` 是代码检查和格式化工具。
- 当前大量 `NotImplementedError` 是正常现象，因为 `tests/adapters.py` 里的函数还没有接到实现。

本模块检查点：

- [x] 能解释 `uv run pytest -q` 每一部分的含义。
- [x] 能解释 `pyproject.toml` 的作用。
- [x] 能说明为什么 `tests/adapters.py` 是测试入口。
- [x] 能从一个失败测试追踪到对应的 adapter 函数。

### 1.1 从失败测试追踪到 adapter

当 pytest 报出 `NotImplementedError` 时，不要先想算法实现，先追踪调用链：

```text
失败测试名
    -> 测试文件中的 test_* 函数
    -> 它调用的 adapters.py 函数
    -> adapters.py 函数的 docstring、参数和返回值
    -> 你在 cs336_basics/ 中的实现
```

建议的读测试顺序：

1. 看 pytest 输出里的第一个失败测试名。
2. 打开对应的 `tests/test_*.py`。
3. 找到这个 `test_*` 函数。
4. 看它从 `tests.adapters` import 了哪个函数。
5. 打开 `tests/adapters.py` 中对应的 `run_*` 或 `get_*` 函数。
6. 只记录输入、输出、shape、边界条件；不要急着写代码。

一个测试失败本质上是在说：

```text
测试给了某些输入
期待 adapter 返回某种输出
但现在 adapter 还没有接到你的实现
```

所以 adapter 是桥，不是最终目的。最终目标是让测试通过这个固定接口调用到你的实现。

下一步：

1. 运行：

   ```sh
   cd /Users/qianmo/assignment1-basics
   uv run pytest tests/test_tokenizer.py -q
   ```

2. 不急着修，先观察第一个失败测试：

   - 测试名是什么？
   - 调用了 `tests/adapters.py` 里的哪个函数？
   - 失败是不是来自 `NotImplementedError`？

结果记录：

- 第一个失败测试：`tests/test_tokenizer.py::test_roundtrip_empty`
- 对应 adapter：`get_tokenizer(...)`
- 失败原因：`NotImplementedError`
- 结论：Module 1 的测试追踪方法已经走通，接下来进入 Tokenizer / BPE。

## Module 2: Tokenizer / BPE

状态：Done

目标：理解文本如何转成 token id，以及如何从语料训练 byte-level BPE tokenizer。

学习顺序修订：

1. Tokenizer 对象接口：`encode` / `decode` / `encode_iterable`。
2. `vocab` / 反向 vocab / `merges` 优先级表。
3. Special token 分割。
4. Pretokenization：普通文本先切成较小片段。
5. BPE inference：对每个 pretoken 的 bytes 按已有 merges 合并。
6. Decode：token ids -> bytes -> UTF-8 字符串。
7. BPE training：从语料训练 `vocab` 和 `merges`。

也就是说，不能跳过 pretokenization。我们只是暂时不做 BPE training；但在 tokenizer inference 中，pretokenization 是必须先理解的步骤之一。

需要先理解的问题：

- 为什么语言模型不能直接处理原始字符串？
- byte-level tokenizer 和 character-level / word-level tokenizer 有什么区别？
- `vocab` 表示什么？
- `merges` 表示什么？
- special tokens 为什么需要特殊处理？
- BPE training 和 tokenizer inference 的区别是什么？
- `encode`、`decode`、`encode_iterable` 分别应该解决什么问题？

相关文件：

- `tests/test_tokenizer.py`
- `tests/test_train_bpe.py`
- `tests/fixtures/gpt2_vocab.json`
- `tests/fixtures/gpt2_merges.txt`
- `tests/fixtures/train-bpe-reference-vocab.json`
- `tests/fixtures/train-bpe-reference-merges.txt`
- `cs336_basics/pretokenization_example.py`

### 2.1 第一个 tokenizer 失败测试

已经观察到的第一个失败：

```text
FAILED tests/test_tokenizer.py::test_roundtrip_empty - NotImplementedError
```

调用链：

```text
test_roundtrip_empty
    -> get_tokenizer_from_vocab_merges_path(...)
    -> tests.adapters.get_tokenizer(vocab, merges, special_tokens)
    -> tokenizer.encode("")
    -> tokenizer.decode(encoded_ids)
```

这个测试还没有要求你训练 BPE。它已经从 GPT-2 fixture 文件中读好了：

- `vocab`: token id 到 token bytes 的映射。
- `merges`: BPE merge 规则，按训练产生的顺序排列。
- `special_tokens`: 可选的特殊 token 列表。

因此，`get_tokenizer(...)` 的职责不是返回 token id，也不是训练 tokenizer，而是返回一个 tokenizer 对象。测试会继续调用这个对象的方法。

从测试可推出的最低接口：

- `encode(text)`: 输入字符串，输出 token id 列表。
- `decode(ids)`: 输入 token id 列表，输出字符串。
- 后续测试还会要求 `encode_iterable(iterable)`。

`test_roundtrip_empty` 的语义：

```text
空字符串
    -> encode
    -> token id 序列
    -> decode
    -> 应该回到空字符串
```

这里的重点不是空字符串怎么特殊处理，而是理解 roundtrip 测试：

```text
decode(encode(text)) == text
```

它验证 tokenizer 至少不会破坏原始文本。

### 2.2 为什么测试里有 get_tokenizer_from_vocab_merges_path

`get_tokenizer_from_vocab_merges_path(...)` 是测试文件里的 helper，不是你必须实现的接口。

它的作用是：

```text
fixture 文件路径
    -> 读取 GPT-2 vocab json
    -> 读取 GPT-2 merges txt
    -> 转成作业要求的 bytes 表示
    -> 调用 adapters.get_tokenizer(vocab, merges, special_tokens)
```

也就是说，path 只和“测试如何加载 fixture 文件”有关。真正的 tokenizer 不应该依赖某个固定文件路径；它只需要依赖已经传进来的 `vocab` 和 `merges` 数据结构。

测试这样设计的好处：

- 测试可以复用 GPT-2 的 vocab/merges 文件作为输入。
- 你的 `get_tokenizer(...)` 接口保持简单，不需要负责读 json/txt 文件。
- tokenizer 可以用任何来源的 vocab/merges 构造，不绑定 GPT-2 文件路径。

职责边界：

```text
get_tokenizer_from_vocab_merges_path: 测试 helper，负责从路径读 fixture。
get_tokenizer: 作业 adapter，负责用 vocab/merges 构造 tokenizer。
tokenizer 对象: 负责 encode/decode/encode_iterable。
```

### 2.3 Tokenizer / BPE 的核心概念

语言模型不能直接处理 Python 字符串。模型的输入通常是整数序列：

```text
text -> token ids -> embedding -> transformer
```

tokenizer 的职责就是在文本和 token id 之间转换：

- `encode(text)`: 字符串到 token id 序列。
- `decode(ids)`: token id 序列回到字符串。
- `encode_iterable(iterable)`: 面向大文本或流式输入，逐段编码并逐个产生 token id。

`vocab` 是词表。这个作业的 adapter 中，`vocab` 的类型是：

```text
dict[int, bytes]
```

也就是：

```text
token id -> token bytes
```

例如某个 id 可能对应字节 `b"hello"`，另一个 id 可能对应字节 `b" "`。注意这里是 bytes，不是 Python 字符串。这样做是 byte-level tokenizer 的关键：它可以覆盖任意 UTF-8 文本，而不需要提前知道所有 Unicode 字符。

`merges` 是 BPE 的合并规则。它的类型是：

```text
list[tuple[bytes, bytes]]
```

每个元素表示训练时学到的一条合并：

```text
(left_token_bytes, right_token_bytes)
```

merges 的顺序很重要。越靠前的 merge 代表越早学到、优先级越高。使用已有 tokenizer 编码文本时，不是在重新训练 merges，而是在根据这份固定规则把 byte 序列合并成更长的 token。

byte-level BPE 可以分成两个不同任务：

```text
BPE training:
    输入语料
    输出 vocab 和 merges

BPE tokenization / inference:
    输入文本、已有 vocab、已有 merges
    输出 token ids
```

当前 `test_tokenizer.py` 主要测第二件事：给定 GPT-2 的 `vocab` 和 `merges`，你的 tokenizer 是否能正确 encode/decode。`test_train_bpe.py` 才会测试第一件事：从语料训练出 vocab 和 merges。

special token 是需要整体保留的字符串，例如 `<|endoftext|>`。它们的原则是：

- 如果 special token 出现在文本里，它应该作为一个整体 token，而不是被普通 BPE 规则拆开。
- 如果传入的 special token 不在 vocab 里，测试 helper 会把它追加到 vocab。
- special token 的处理发生在普通 BPE tokenization 之前。

本阶段建议先把 tokenizer inference 和 BPE training 分开学：

1. 先理解如何使用已有 `vocab` / `merges` 做 encode/decode。
2. 再理解如何从语料统计并训练出 `merges`。

注意：第一步 tokenizer inference 里仍然包含 special token 分割和 pretokenization。

### 2.4 Pretokenization 是什么

pretokenization 是 BPE merge 之前的一步。它把一整段字符串先切成较小的文本片段，然后每个片段再进入 byte-level BPE。

整体流程可以理解为：

```text
原始字符串
    -> special token 分割
    -> pretokenization 切成文本片段
    -> 每个片段转成 UTF-8 bytes
    -> 从单字节 token 开始按 merges 合并
    -> 查 vocab 得到 token ids
```

为什么需要 pretokenization：

- 限制 BPE merge 的作用范围。通常不希望 BPE 在任意字符之间无限跨越，比如跨越很多单词或特殊边界。
- 保留类似 GPT-2 tokenizer 的行为。GPT-2 不是直接对整篇文本的一长串 bytes 做 BPE，而是先用正则切成类似词、空格、标点、数字片段。
- 提高效率。短片段上的 BPE 合并比整篇文本一起处理更容易控制。

pretokenization 和 BPE 的区别：

```text
pretokenization:
    字符串级别的粗切分，决定哪些片段分别处理。

BPE:
    bytes/token 级别的细合并，把短片段压成更长 token。
```

作业里的 `cs336_basics/pretokenization_example.py` 不是完整 tokenizer。它只展示了另一类“预处理”：训练 BPE 时如何按特殊 token 边界把大文件切成多个 chunk，方便并行统计。这个文件里的 chunk boundary 和 tokenizer 中的 pretokenization 相关，但不是同一层职责：

```text
find_chunk_boundaries:
    大文件 -> 多个可独立处理的 chunk

pretokenization:
    单个字符串/chunk -> 多个可进行 BPE 的文本片段
```

### 2.5 `matches_tiktoken` 测试是什么意思

`tests/test_tokenizer.py` 中名字带 `matches_tiktoken` 的测试，是把你的 tokenizer 输出和 OpenAI `tiktoken` 里的 GPT-2 tokenizer 输出进行比较。

测试流程通常是：

```text
reference_tokenizer = tiktoken.get_encoding("gpt2")
tokenizer = get_tokenizer_from_vocab_merges_path(gpt2_vocab, gpt2_merges, ...)

reference_ids = reference_tokenizer.encode(text, ...)
ids = tokenizer.encode(text)

assert ids == reference_ids
```

这些测试成立的前提是：测试 helper 给你的 `vocab` 和 `merges` 来自 GPT-2 fixture。因此这里不是要求你的任意 tokenizer 都等于 GPT-2，而是要求：

```text
当输入 GPT-2 的 vocab/merges 时，你的 BPE tokenizer 行为应当和 GPT-2 tokenizer 一致。
```

这类测试主要验证：

- pretokenization 是否和 GPT-2 兼容。
- byte-level 编码是否正确。
- BPE merge 优先级是否正确。
- special token 处理是否正确。
- decode 是否能还原同样文本。

### 2.6 Tokenizer 对象内部应该保存什么

`get_tokenizer(vocab, merges, special_tokens)` 的返回值应该是一个 tokenizer 对象。这个对象不是一次性函数，因为 encode、decode、encode_iterable 都需要共享同一组 tokenizer 状态。

对象内部至少需要保存这些信息：

| 内部状态 | 来源 | 用途 |
| --- | --- | --- |
| `vocab` | adapter 传入 | decode 时从 token id 找到 token bytes |
| 反向 vocab | 由 `vocab` 反转得到 | encode 时从 token bytes 找到 token id |
| merge priority | 由 `merges` 转换得到 | BPE 合并时判断哪个 pair 优先级最高 |
| special tokens | adapter 传入 | encode 时把特殊字符串作为整体 token 保留 |

为什么需要反向 vocab：

```text
decode:
    token id -> token bytes

encode:
    token bytes -> token id
```

传入的 `vocab` 只支持第一种方向，所以 encode 需要准备反向查表。

为什么需要 merge priority：

`merges` 是一个有顺序的列表，顺序代表优先级。BPE inference 时经常需要判断某个相邻 pair 在 merges 里的排名。如果每次都在列表里线性查找，会很慢。因此通常会把它转换成适合快速查优先级的数据结构。

special tokens 的保存方式要支持两个目标：

- 能判断某段文本是不是 special token。
- 如果多个 special token 有重叠，要优先匹配正确的整体 token。

设计边界：

- tokenizer 对象可以保存数据和方法。
- `get_tokenizer(...)` 只负责构造 tokenizer 对象。
- 文件路径读取是测试 helper 的职责，不属于 tokenizer 对象的核心职责。

### 2.7 `encode(text)` 的概念流程

`encode(text)` 的目标是把一个 Python 字符串转换成 token id 列表。

推荐从流程上理解：

```text
输入 text
    -> 按 special tokens 切分
    -> special token 片段直接查 id
    -> 普通文本片段做 pretokenization
    -> 每个 pretoken 转 UTF-8 bytes
    -> 从单字节 token 序列开始
    -> 反复按 merges 优先级合并相邻 token
    -> 每个最终 bytes token 查反向 vocab
    -> 输出 token ids
```

每一步的职责：

1. Special token 分割

   如果文本中有 `<|endoftext|>` 这类 token，先把它们分离出来。special token 不进入普通 pretokenization，也不进入普通 BPE merge。

2. Pretokenization

   对普通文本片段做字符串级别的粗切分。GPT-2 风格 tokenizer 会把文本切成类似词、空格前缀词、数字、标点、空白片段。

3. Byte encoding

   每个 pretoken 先变成 UTF-8 bytes。byte-level tokenizer 的基础 token 是单个 byte，因此任意文本都能表示。

4. BPE merge

   对每个 pretoken 的 byte token 序列，根据 `merges` 的优先级逐步合并。注意 merge 只发生在当前 pretoken 内部，不跨 pretoken 边界，也不跨 special token 边界。

5. 查 token id

   BPE 合并结束后，每个 bytes token 都应该能在反向 vocab 中查到 id。

容易出错的地方：

- 忘记 special token 要整体保留。
- BPE 在不同 pretoken 之间跨边界合并。
- merge 优先级处理错，把“最早出现的 pair”误认为“最高优先级 pair”。
- bytes 和 str 混用。
- 空字符串应该编码为空列表。

### 2.8 Tokenizer 实现拆分路线

`encode` 不适合一次写完。建议把 tokenizer 拆成几个可以单独检查的小任务：

1. 构造 tokenizer 对象

   目标：`get_tokenizer(vocab, merges, special_tokens)` 能返回一个对象。对象内部保存：

   - `vocab`: id -> bytes
   - `inverse_vocab`: bytes -> id
   - `merge_priority`: pair -> rank
   - `special_tokens`: 需要整体保留的字符串

2. 先做 `decode(ids)`

   目标：给一串 token id，按顺序查 `vocab`，把 bytes 拼起来，再解码成字符串。

   先检查：

   ```text
   decode([]) == ""
   decode([某个单字节 token id]) == 对应字符
   ```

3. 再做最小 `encode(text)` 的外壳

   目标：先能处理空字符串，返回空 token id 列表。

   这一步只为了让第一个 roundtrip 测试往前走。

4. 实现普通文本的 pretokenization

   目标：把普通字符串切成 GPT-2 风格的 pretoken。先不要处理 special token，也不要急着做 BPE training。

5. 实现“单个 pretoken 的 BPE”

   目标：输入一个 pretoken 的 bytes，输出若干最终 bytes token。

   检查点：

   - 初始状态是一个个单 byte token。
   - 每次只合并当前序列里优先级最高的相邻 pair。
   - 如果当前没有 pair 出现在 merge 表里，就停止。

6. 把普通文本 encode 串起来

   目标：pretokenization -> BPE -> 反向 vocab 查 id。

7. 加 special token

   目标：special token 作为整体保留，不进入普通 pretokenization 和 BPE。

8. 加 `encode_iterable`

   目标：逐段处理 iterable，逐个产生 token id。注意它有内存测试，所以不能把所有内容一次性拼成一个巨大字符串。

建议验证顺序：

```text
decode 空输入
-> encode 空字符串
-> 单字符 roundtrip
-> 单字符 matches_tiktoken
-> ASCII 字符串
-> Unicode 字符串
-> special token
-> encode_iterable
```

进度记录：

- `tests/test_tokenizer.py::test_roundtrip_empty` 已通过。
- 已确认 `tests.adapters.get_tokenizer(...) -> BPETokenizer -> encode("") -> decode([])` 这条链路可用。
- 下一步目标：支持单字符普通文本的 `encode`，并对齐 `test_roundtrip_single_character` / `test_single_character_matches_tiktoken`。
- 学习节奏修订：先完整学习 tokenizer/BPE 知识，再继续实现。当前讲解从 handout Section 2.1-2.3 开始：Unicode、UTF-8 encoding、subword tokenization。

### 2.9 Decode 实现注意点

`decode(ids)` 的中间结果应该保持为 `bytes`，最后再统一转成 `str`。

正确的方向：

```text
ids
    -> 用 vocab 查出多个 bytes 片段
    -> 拼成一个完整 bytes
    -> 按 UTF-8 解码成 Python str
```

常见错误：

```text
str(bytes_obj)
```

这不会把 bytes 内容按 UTF-8 解码，而是得到 bytes 对象的字符串表示。例如概念上会变成类似：

```text
"b'hello'"
```

真正需要的是 bytes 的解码操作，而不是 `str(...)` 包装。

另外，循环变量不要命名成 `id`。`id` 是 Python 内置函数名，虽然不会立刻报错，但会让代码可读性变差。可以用 `token_id`。

### 2.10 Tokenizer/BPE 系统讲解

学习目标：先理解 tokenizer 的完整数据流，再写代码。

#### 2.10.1 Tokenizer 解决什么问题

语言模型不能直接吃 Python 字符串。它需要整数 token id：

```text
text
    -> tokenizer
    -> token ids
    -> embedding
    -> transformer
```

tokenizer 是文本和整数之间的桥：

```text
encode: str -> list[int]
decode: list[int] -> str
```

#### 2.10.2 为什么是 byte-level

如果直接按词切分，会遇到没见过的新词。如果直接按 Unicode 字符建 vocab，字符空间很大，而且组合复杂。

byte-level 的想法是：

```text
任意 Unicode 字符串
    -> UTF-8 bytes
    -> 每个 byte 都在 0..255 范围内
```

所以初始 vocab 只需要覆盖 256 个单字节 token，就永远不会有 out-of-vocabulary 的普通文本。

缺点是序列会比较长。BPE 的作用就是把常见相邻 byte 序列合并成更长 token，从而压缩序列长度。

#### 2.10.3 BPE training 和 tokenizer inference

两件事必须分开：

```text
BPE training:
    输入训练语料
    学出 vocab 和 merges

Tokenizer inference:
    输入文本 + 已有 vocab/merges
    输出 token ids
```

当前 `test_tokenizer.py` 主要测 inference。`test_train_bpe.py` 才测 training。

#### 2.10.4 vocab 和 merges

`vocab` 是 token id 到 bytes 的映射：

```text
id -> bytes
```

decode 用它。

encode 需要反方向：

```text
bytes -> id
```

所以 tokenizer 初始化时通常会构造反向 vocab。

`merges` 是训练阶段学出来的合并历史：

```text
[(left_bytes, right_bytes), ...]
```

顺序代表优先级。编码时要按这个优先级把小 token 合成大 token。

#### 2.10.5 Pretokenization 的位置

pretokenization 是字符串级别的粗切分。它发生在 BPE merge 之前：

```text
text
    -> special token 分割
    -> pretokenization
    -> pretoken bytes
    -> BPE merges
    -> ids
```

它的作用：

- 限制 merge 只在合理片段内部发生。
- 不让 BPE 任意跨词、标点或特殊边界合并。
- 让训练时可以统计 pretoken 频率，而不是每轮扫描整篇语料。

作业采用 GPT-2 风格 regex pretokenizer。重要直觉是：空格经常会跟后面的词形成同一个 pretoken，例如 `" world"`。

#### 2.10.6 Encode 的数据流

`encode(text)` 的概念流程：

```text
输入 str
    -> 找出 special token spans
    -> 普通文本片段分别 pretokenize
    -> 每个 pretoken 转 UTF-8 bytes
    -> 初始表示为单字节 bytes token 序列
    -> 根据 merges 把相邻 token 合并
    -> 每个最终 bytes token 查反向 vocab
    -> 输出 list[int]
```

关键边界：

- 不跨 pretoken 边界 merge。
- 不跨 special token 边界 merge。
- special token 直接作为整体查 id。
- 普通文本从 UTF-8 bytes 开始，因此任意 Unicode 都能处理。

#### 2.10.7 单个 pretoken 的 BPE

对一个 pretoken 来说，BPE inference 的核心是：

```text
初始: [single_byte_1, single_byte_2, ...]
反复:
    找当前相邻 pair 中 merge 优先级最高的 pair
    把所有/对应出现位置合并成 left+right
直到没有可合并 pair
```

从作业 handout 的角度，也可以理解为：拿训练时得到的 merges，按创建顺序应用到当前 pretoken。

两种说法等价的前提是优先级处理正确：

```text
越早出现在 merges 中的 pair，rank 越小，优先级越高。
```

#### 2.10.8 Decode 的数据流

decode 比 encode 简单：

```text
ids
    -> vocab[id] 得到 bytes pieces
    -> 拼成完整 bytes
    -> UTF-8 decode 成 str
```

注意不要逐 token 先 decode 成字符串再拼，因为一个 Unicode 字符的 bytes 可能被多个 token 分开。

如果 bytes 不是合法 UTF-8，作业 handout 要求用 replacement character 处理 malformed bytes。

#### 2.10.9 Special tokens

special token 是用户指定的特殊字符串，例如 `<|endoftext|>`。

原则：

- 它必须作为一个整体 token。
- 它不参与普通 BPE merge。
- 它会形成硬边界，防止左右两侧普通文本跨过去 merge。
- 如果多个 special token 重叠，要优先让更长或更具体的 special token 保持整体。

训练时，special token 用于分割语料，但不贡献普通 merge 统计。编码时，special token 本身会输出对应 id。

#### 2.10.10 encode_iterable

`encode(text)` 可以一次性处理一个字符串；`encode_iterable(iterable)` 用于大文件或流式输入。

目标：

```text
Iterable[str] -> lazily yield token ids
```

核心要求是内存稳定，不能把整个大文件读进一个巨大字符串。但还要小心：如果随意按 chunk 单独 encode，可能改变跨 chunk 边界的 tokenization。因此需要设计 chunk 边界，或保留足够上下文，避免 BPE token 跨边界时结果不一致。

测试中会对 `encode_iterable` 做 roundtrip、与 tiktoken 对齐，以及内存行为检查。

#### 2.10.11 当前实现顺序

接下来写代码时按这个顺序：

1. `decode` 完整化：bytes join 后用 UTF-8 decode，并处理 malformed bytes。
2. 普通文本 pretokenization。
3. 单个 pretoken 的 BPE merge。
4. 普通文本 encode：pretokenize -> BPE -> id。
5. special token 分割和重叠 special token。
6. `encode_iterable`。
7. 再进入 BPE training。

实现决策：

- `encode(text)` 不应该直接对整段文本运行 BPE。
- 如果配置了 special tokens，先按 special token 把文本切成普通片段和 special token 片段。
- special token 片段直接映射成对应 id。
- 普通片段再做 pretokenization。
- 每个 pretoken 内部独立运行 BPE，不跨 pretoken 边界合并。

### 2.11 Tokenizer 课堂版摘要

Tokenizer 是语言模型前面的文本接口：

```text
原始文本 -> token ids -> embedding -> transformer
```

本作业里的 tokenizer 是 byte-level BPE tokenizer。它先把文本表示成 UTF-8 bytes，保证任意 Unicode 文本都能表示；再用 BPE merges 把常见 byte 序列合并成更长 token，减少序列长度。

四个核心概念：

- `vocab`: token id 到 token bytes 的映射，主要用于 decode。
- `reverse_vocab`: token bytes 到 token id 的映射，主要用于 encode。
- `merges`: BPE 合并规则，顺序代表优先级。
- `pretokenization`: BPE 前的字符串级粗切分，限制 merge 的边界。

Tokenizer inference 的 encode 流程：

```text
text
    -> special token 分割
    -> 普通片段 pretokenize
    -> pretoken 转 UTF-8 bytes
    -> 单字节 token 序列
    -> 按 merges 做 BPE 合并
    -> reverse_vocab 查 id
    -> list[int]
```

Decode 流程：

```text
list[int]
    -> vocab 查 bytes
    -> 拼接 bytes
    -> UTF-8 decode
    -> str
```

Training 和 inference 的区别：

```text
BPE training:
    从语料统计并生成 vocab/merges

Tokenizer inference:
    使用已有 vocab/merges 编码新文本
```

当前阶段先完成 inference，再做 training。

### 2.12 Special token 的处理

special token 是用户指定的特殊字符串，例如 `<|endoftext|>`。它表达的是元信息，不是普通自然语言片段。

在 tokenizer inference 的 `encode(text)` 中，special token 的处理流程是：

```text
输入 text
    -> 找出 special token
    -> 把文本切成 special token 片段和普通文本片段
    -> special token 片段直接转成对应 token id
    -> 普通文本片段走 pretokenization + BPE
    -> 按原始顺序拼接所有 token ids
```

识别之后，special token 不应该再进入普通流程：

```text
special token
    -> special_token.encode("utf-8")
    -> reverse_vocab 查 id
    -> 直接输出这个 id
```

普通文本片段仍然走：

```text
普通文本
    -> pretokenization
    -> UTF-8 bytes
    -> BPE merge
    -> reverse_vocab 查 ids
```

关键边界：

- special token 必须作为一个整体 token。
- BPE 不能在 special token 内部合并。
- BPE 不能跨 special token 左右边界合并。
- 如果 special tokens 重叠，应优先匹配更长的 special token。

例子：

```text
"A<|endoftext|>B"
```

应被看成三段：

```text
"A"               普通文本
"<|endoftext|>"   special token
"B"               普通文本
```

输出 ids 的顺序仍然对应原文顺序：

```text
encode("A") + [id_of_endoftext] + encode("B")
```

在 decode 中，不需要特别识别 special token。因为 special token 已经在 vocab 里有对应 bytes：

```text
id -> b"<|endoftext|>" -> 拼 bytes -> decode
```

在 BPE training 中，special token 的职责不同：

- 它们作为硬边界切分语料。
- 它们不参与普通 pair frequency 统计。
- 它们应该被加入 vocab，拥有固定 id。

当前实现步骤：

- 先写一个 special token 切分层。
- 输入：完整 `text`。
- 输出：保持原文顺序的一串片段，每个片段要能区分“普通文本”还是“special token”。
- 没有 special token 时，整段文本就是一个普通片段。
- special token 片段后续直接查 id。
- 普通片段后续进入 pretokenization。
- 如果 special tokens 有重叠，要优先匹配更长的 special token。

实现思路：

- 方案 A：用 regex 找 special token。把所有 special token 转义后拼成一个“匹配任意 special token”的模式，用匹配位置把原文切开。
- 方案 B：手动扫描字符串。在当前位置检查是否有 special token 匹配；若有，输出 special token 片段并向前跳；若没有，继续累积普通文本。
- 两种方案都要先处理重叠 token：把 special tokens 按长度从长到短排序，保证更长 token 优先。
- 输出的数据最好带 tag，例如概念上的 `(is_special, piece)`，否则后面不知道哪段应该直接查 id，哪段应该继续 pretokenize。
- 自测时先覆盖：没有 special token、special token 在开头、在结尾、连续出现、重叠 token、special token 两侧为空字符串。

学习补充：

- 用 toy marker 练习 special token 切分，不直接从作业 token 开始。
- 需要掌握 `re.escape(marker)`：把 marker 中的正则特殊字符转义。
- 需要掌握 `re.finditer(pattern, text)`：逐个得到匹配对象。
- 匹配对象的 `start()` / `end()` 可以用来切出 match 前后的普通文本。
- 用 `text[last:match.start()]` 取得上一个匹配结束后到当前匹配开始前的普通片段。
- 用 `match.group()` 取得匹配到的 marker 本身。

### 2.13 怎么做 pretokenization

pretokenization 的输入是普通文本片段，输出是一串 pretoken 字符串。它只负责字符串级粗切分，不负责 BPE merge，也不负责查 token id。

在本作业里，pretokenization 应该使用 handout 给出的 GPT-2 风格 regex pattern：

```python
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

注意要用第三方 `regex` 包，而不是 Python 内置的 `re`，因为 `\p{L}` 和 `\p{N}` 是 Unicode character property。

这个 pattern 的几个分支：

| 分支 | 大致含义 |
| --- | --- |
| `'(?:[sdmt]|ll|ve|re)` | 英文缩写后缀，如 `'s`, `'t`, `'ll`, `'re` |
| ` ?\p{L}+` | 可选空格 + 连续字母 |
| ` ?\p{N}+` | 可选空格 + 连续数字 |
| ` ?[^\s\p{L}\p{N}]+` | 可选空格 + 连续标点/符号 |
| `\s+(?!\S)` | 一段后面不是非空白字符的空白，常用于尾部空白 |
| `\s+` | 其他空白 |

读法重点：

- 很多分支前面有一个可选空格 ` ?`。
- 这就是为什么 GPT-2 风格 tokenizer 经常产生 `" hello"` 这种带前导空格的 pretoken。
- pretokenization 保留空白信息，因为 byte-level tokenizer 需要能完整还原原文。

示例：

```text
"Hello, how are you?"
    -> "Hello", ",", " how", " are", " you", "?"
```

另一个示例：

```text
"some text that i'll pre-tokenize"
    -> "some", " text", " that", " i", "'ll", " pre", "-", "tokenize"
```

与 special token 的顺序：

```text
先按 special token 切分文本
    -> special token 片段直接查 id
    -> 普通文本片段才运行 PAT pretokenization
```

不要让 special token 进入普通 pretokenizer，否则它可能被拆成多个普通片段。

实现检查点：

- pretokenizer 输入和输出都是 `str`。
- 对普通文本使用 `regex.finditer` 逐个产生 match，避免不必要地保存大列表。
- pretokenizer 不做 UTF-8 bytes 转换；bytes 转换是下一步 BPE inference 的职责。
- pretokenizer 不做 BPE merge；它只决定 merge 的边界。

### 2.14 BPE merge 时如何表示“已经合并”

BPE 不应该直接在原始 `str` 上做标记。正确的思路是维护一个“当前 token 序列”。

对于一个 pretoken：

```text
"the"
    -> UTF-8 bytes
    -> 当前 token 序列: [b"t", b"h", b"e"]
```

如果合并 `(b"t", b"h")`，不是在字符串 `"the"` 上画标记，而是把当前序列里的两个相邻元素替换成它们拼接后的 bytes：

```text
[b"t", b"h", b"e"]
    -> [b"th", b"e"]
```

如果继续合并 `(b"th", b"e")`：

```text
[b"th", b"e"]
    -> [b"the"]
```

所以“标记已合并”的方式就是：当前 token 序列发生了变化。两个旧 token 消失，一个新 token 出现。

关键点：

- 当前 token 序列里的每个元素都是 `bytes`。
- 相邻 pair 来自当前 token 序列，而不是原始字符串。
- 每次 merge 后，要基于新的 token 序列重新看相邻 pair。
- merge 的结果是 `left + right`，仍然是一个 `bytes` token。

补充：不要混淆“重新从 merges 开始找规则”和“只合并一个位置”。

对 tokenizer inference 来说，常见实现是：

```text
while 当前 token 序列还有可合并 pair:
    找当前序列中 merge rank 最高的 pair
    把这个 pair 在当前序列中的不重叠出现从左到右合并
    基于新 token 序列重新找可合并 pair
```

handout 中“回到 merge list 查找下一条可用 merge”的意思是：每次 token 序列更新后，要重新按 merge 优先级判断下一条规则。它不等于“每次只合并一个位置就立刻停止这一轮”。

原始 BPE training 通常也可以理解为：每轮选一个最高频 pair，然后替换语料中该 pair 的出现。encoding/inference 则使用训练得到的 merge 顺序作为优先级。

### 2.15 当前 tokenizer 草稿审查记录

已经开始写 `BPETokenizer`，并完成了最小 `decode` 和部分 `encode` / merge 草稿。

当前需要修正的关键点：

- `special_tokens` 可能是 `None`，排序或遍历前要先处理成空列表。
- special token 切分应作用在原始字符串或普通文本字符串上，不能先 pretokenize 成 list 后再对 list 运行 regex。
- `merge` 中当前 token 序列应该是 `list[bytes]`，不是 `list[str]`。
- BPE merge 循环里需要初始化扫描下标，并在每轮/每条 merge 后重置临时结果。
- 每次合并后，必须用新的 token 序列继续下一轮判断。
- 普通片段传给 `merge` 时应该是单个 pretoken 字符串；如果传入 list，需要先明确它代表多个 pretokens 并逐个处理。
- `decode` 应使用 `errors="replace"` 处理非法 UTF-8 token id 序列。
- 避免用 `id`、`i` 这种容易混淆或覆盖内置含义的变量名。

第二版审查：

- `encode` 中 special token 切分已提前到 pretokenization 之前，方向正确。
- pretokenization 分支写反了：应该对普通文本片段运行 PAT；special token 片段不运行 PAT。
- 当前 pretokenization 得到的 `tokens` 列表没有被后续 tokenization 使用，属于死代码。
- `strList` 的类型标注应表达 piece 是 `str`，不是 `list[str]`。
- 如果没有 special tokens，`pat` 为空会导致正则匹配行为异常；应先走普通文本路径。
- `merge` 内部仍需修正：初始化扫描下标、每轮重置 `new_tokens`、合并后更新当前 token 序列。

第三版审查：

- `merge` 的输入被改成 `list[str]`，但 encode 里实际传入的仍是普通文本片段 `str`；函数边界需要重新定死。
- `merge` 最清楚的职责应该是“单个 pretoken 字符串 -> token ids”。
- `tokens` 变量名被重复用于不同含义，导致逻辑难以检查。建议区分 `piece`、`pretoken`、`byte_tokens`、`ids`。
- pretokenization 得到的 `tokens` 仍然没有参与后续流程。
- BPE 当前序列应始终是 `list[bytes]`，不要标成 `list[str]`。
- special token 为空时仍需要显式走无 special token 路径。

第四版审查：

- `strList` 类型改成 `list[tuple[bool, str]]` 是正确方向。
- `self.special_tokens.empty()` 不是 Python list 的方法；判断列表是否为空应使用 truthiness。
- pretokenization 结果 `tokens` 是 `list[str]`，但后续循环写成了 `for is_special, pretoken in tokens`，会错误地把字符串当作二元组解包。
- special token 片段在 pretokenization 阶段被丢失了；不能只把普通 pretoken 放进 `tokens` 后再统一 tokenization。
- 正确结构仍应是：遍历 `strList`，special 直接查 id，普通 piece 内部再遍历 PAT matches。
- `merge` 中每轮需要创建本轮 `next_tokens`，合并后更新 `byte_tokens`；不能一直往初始化用的 `new_tokens` 里追加。
- `merge` 的 `else` 分支仍缺少位置前进。

第五版审查：

- `merge` 的输入改成单个 `string: str`，方向正确。
- `encode` 的无 special token 分支写反：`else` 中又判断 `if self.special_tokens`，导致 `None` 时不会加入普通文本片段。
- pretokenization 仍然先收集到全局 `tokens: list[str]`，随后又按 `(is_special, pretoken)` 解包，结构不匹配。
- special token 片段仍没有进入最终 tokenization 流程。
- `merge` 中 `new_tokens` 应该在每次合并某个 pair 时重新创建；当前在 while 外创建，会持续累积旧结果。
- `merge` 合并后仍没有执行 `bytes_tokens = new_tokens`，因此当前 token 序列不会更新。
- `merge` 当前按 `self.merges` 顺序扫描可工作在概念上接近 handout，但需要在应用一次 merge 后重新从当前新序列继续，而不是在同一轮不断往旧 `new_tokens` 追加。

第六版审查：

- `encode` 的 `None` special token 分支已修正：没有 special token 时会把整段文本作为普通片段。
- 仍需处理 `special_tokens=[]` 的情况；空列表不应构造空 regex pattern。
- `tokens` 是 `list[str]`，但后续写成 `for is_special, pretoken in tokens`，结构不匹配。不要把 pretoken 列表当作 `(is_special, piece)` 列表。
- 当前 special token 片段仍会在 pretokenization 后的全局 `tokens` 列表中丢失。
- 应把 pretokenization 放进 `for is_special, piece in strList` 的普通文本分支里，而不是先生成全局 `tokens`。
- `merge` 仍需要更新当前序列：每次应用某个 merge 后，用本轮结果替换当前 `bytes_tokens`。
- `merge` 的本轮结果列表不能在 while 外长期累积。

第七版审查：

- `special_tokens is not None and special_tokens != []` 已覆盖 `None` 和空列表，方向正确；后续可简化为初始化时统一成列表。
- `merge` 中已有 `bytes_tokens = new_tokens` 和 `new_tokens = []`，但最终 ids 却从已经清空的 `new_tokens` 构造，应该检查最终 token 序列到底保存在哪个变量中。
- `encode` 中 pretokenization 结果 `tokens` 仍未使用；最终仍然把普通文本 piece 整段传给 `merge`，会允许跨 pretoken 边界合并。
- 正确数据流仍是：遍历 `strList`；special 直接查 id；普通 piece 内部用 PAT 逐个得到 pretoken，并对每个 pretoken 调 `merge`。
- `merge` 中变量 `merge` 与方法名接近，建议改成 `merge_pair` 或类似名字，减少读代码时的歧义。

第八版审查：

- `encode` 主流程已明显改善：普通文本片段会经过 PAT，且每个 `match.group()` 会进入 `merge`。
- special token 分支应产生单个 id；对单个整数不能用 `extend`。
- `merge` 的主要问题是把“遍历 merges 规则”和“生成本轮新 token 序列”混在一起。当前代码会对每条 merge 规则都把当前 token 序列复制/追加到 `new_tokens`，即使没有发生合并。
- 用单字符 `"s"` 思考：当前序列长度为 1 时，内层 while 不运行，但对每条 merge 规则都会追加这个 token，结果会产生大量重复 token。
- BPE inference 应该是：当前序列中找一条可应用的最高优先级 merge；如果找不到就停止；如果找到，基于当前序列生成一次新序列并替换当前序列。
- `decode` 后续仍需改为 malformed UTF-8 使用 replacement，并避免使用 `id` 作循环变量。

当前修复任务：

- 在 `encode` 的 special token 分支，把“加入一个整数 id”和“加入多个 id”区分开。
- 在 `merge` 中先得到初始 `byte_tokens`。
- 每一轮先找 `byte_tokens` 当前相邻 pairs 中是否有可应用的 merge 规则。
- 找到一条最高优先级规则后，再单独扫描一次 `byte_tokens` 生成 `next_tokens`。
- 如果找不到可应用规则，停止循环。
- 循环结束后，从最终的 `byte_tokens` 构造 ids。

第九版测试记录：

运行 `uv run pytest tests/test_tokenizer.py -q` 后出现 16 failed、7 passed、2 skipped。

主要根因不是 16 个独立问题，而是：

- 多数 roundtrip 测试在 tokenizer 构造阶段失败：`TypeError: 'NoneType' object is not iterable`。实际运行文件的 `__init__` 中仍有 `sorted(special_tokens, ...)` 直接处理 `None` 的代码，需要先修这个根因。
- `test_roundtrip_unicode_string_with_special_tokens` 中，`decode([127])` 报 `UnicodeDecodeError`。这是因为单个 token 可能只是一个多字节 UTF-8 字符的一部分，`decode` 要使用 malformed bytes replacement。
- `encode_iterable` 尚未实现，因此 iterable 相关测试失败是预期的后续任务。

排查顺序：

1. 先确认磁盘上的 `cs336_basics/tokenizer.py` 和当前粘贴代码一致，尤其是 `__init__` 第 7 行。
2. 修 `special_tokens=None` 初始化。
3. 修 `decode` 的 UTF-8 error handling。
4. 跑最小 tokenizer tests，再处理 `encode_iterable`。

第十版测试记录：

再次运行 tokenizer 全量测试后仍有 16 failed、7 passed、2 skipped。

当前最优先根因：

- `cs336_basics/tokenizer.py` 第 8 行实际逻辑是：当 `special_tokens is None` 时仍然调用 `sorted(special_tokens, ...)`。这是条件分支写反导致的。
- 这会导致所有没有传 special token 的测试都在 tokenizer 构造阶段失败，后续 encode/decode 根本没跑到。
- `test_overlapping_special_tokens` 失败说明重叠 special token 仍被拆成了多个短 token，后续需要检查 special token 排序和 regex alternation 顺序。
- `encode_iterable` 尚未实现，仍是后续任务。

当前排查原则：

1. 先修 `special_tokens=None` 初始化，直到 `test_roundtrip_empty` 能进入 encode/decode。
2. 再看普通 encode/decode。
3. 再修 overlapping special tokens。
4. 最后实现 `encode_iterable`。

第十一版测试记录：

运行 `uv run pytest tests/test_tokenizer.py -q` 后结果为 21 passed、2 failed、2 skipped。

已通过：

- empty / single character / unicode / ascii / special token / overlapping special token / file fixture roundtrip / tiktoken 对齐等 tokenizer 主体测试。

剩余失败：

- `test_encode_iterable_tinystories_sample_roundtrip`
- `test_encode_iterable_tinystories_matches_tiktoken`

根因：

- `BPETokenizer` 还没有实现 `encode_iterable` 方法。

下一步：

- 先实现功能正确的 `encode_iterable(iterable)`，让它能遍历 file handle 的每个 string chunk，并逐个 yield token id。
- 当前 macOS 上内存测试被 skipped；后续再按 handout 的 memory considerations 优化边界和内存行为。

第十二版测试记录：

- 用户实现 `encode_iterable` 后，`tests/test_tokenizer.py` 中除 macOS skipped 的两个 memory tests 外均已通过。
- 当前实现方式是逐 chunk 调用 `encode`。这能通过现有功能测试，主要因为测试传入的是 Python file handle，默认按行迭代；该 fixture 下行边界没有触发可见的不一致。
- handout 的 Memory considerations 明确提醒：处理大文件时要分 chunk，但必须保证 token 不跨 chunk 边界，否则 tokenization 可能不同于一次性编码完整文本。
- 因此：当前 tokenizer 功能测试可视为完成；从严格规格看，`encode_iterable` 仍有一个可选优化点，即用 buffer 保留可能跨 chunk 边界的尾部。
- macOS 上 memory tests 被 skip 是因为测试文件用 `skipif(not sys.platform.startswith("linux"))` 跳过；这不表示要求无效。提交/评分环境如果是 Linux，memory tests 可能会运行。因此最终提交前应重新检查 `encode_iterable` 的内存行为，至少不能一次性读完整文件。

### 2.15.1 encode_iterable 的 chunk 边界问题

`encode_iterable` 的目标不是简单地把大文件拆开独立编码，而是：

```text
list(tokenizer.encode_iterable(iterable)) == tokenizer.encode(完整文本)
```

同时又不能为了这个等式把完整大文件一次性读入内存。

为什么不能无脑逐 chunk 调 `encode(chunk)`：

```text
完整文本: "hello"
chunk 形式: "he" + "llo"
```

如果两个 chunk 分别 encode，BPE 只能在 `"he"` 内部和 `"llo"` 内部合并；但完整 encode 时，BPE 可以在整个 pretoken `"hello"` 内部合并。最终 token ids 可能不同。

special token 也有类似问题：

```text
完整文本: "<|endoftext|>"
chunk 形式: "<|endo" + "ftext|>"
```

如果每个 chunk 独立处理，就识别不出完整 special token。

因此，严格正确的流式 tokenizer 需要保留 buffer：

1. 从 iterable 读入一个 chunk，追加到 buffer。
2. 只编码“确定不会再被下一个 chunk 改变”的前缀。
3. 把可能跨 chunk 边界的尾部留在 buffer 里。
4. 文件结束后，再编码剩余 buffer。

BPE 的好消息是：merge 不会跨 pretoken 边界。因此一个可行的思路是，只输出已经完整确定的 pretokens，把最后一个可能还会延伸的 pretoken 留到下一轮。

现阶段理解重点：

- `encode_iterable` 应该是 generator，用 `yield` 逐个产出 token id。
- 简单逐行 `encode(line)` 占用内存小，但不保证和完整 `encode` 完全一致。
- 一次性 `encode("".join(iterable))` 功能上最简单，但不满足大文件内存目标。
- 严格版本需要在“功能正确”和“内存稳定”之间通过 buffer 边界处理达成一致。

### 2.16 BPE inference 方式 B：扫描当前 pair 查 rank

方式 B 的核心思想：

```text
不要每轮扫描完整 merges 列表。
只扫描当前 byte_tokens 中真实存在的相邻 pair。
每个 pair 去 merge_rank 表里查优先级。
选择 rank 最小的 pair 作为本轮要合并的 pair。
```

需要的数据结构：

- `merge_rank`: 由 `merges` 构造，表示 `pair -> rank`。
- `byte_tokens`: 当前 token 序列，类型是 `list[bytes]`。
- `best_pair`: 当前这一轮要合并的 pair。
- `best_rank`: 当前找到的最小 rank。
- `next_tokens`: 应用 `best_pair` 后生成的新序列。

每轮分两步：

1. 找 best pair

   遍历当前相邻 pair：

   ```text
   (byte_tokens[0], byte_tokens[1])
   (byte_tokens[1], byte_tokens[2])
   ...
   ```

   如果 pair 在 `merge_rank` 中，就比较它的 rank。rank 越小，优先级越高。

2. 应用 best pair

   如果本轮找不到 best pair，BPE 结束。

   如果找到了，就从左到右扫描 `byte_tokens`，把所有不重叠的 `best_pair` 合并成 `left + right`，生成 `next_tokens`。然后令当前序列变成 `next_tokens`，进入下一轮。

边界检查：

- 长度为 0 或 1 的 `byte_tokens` 没有相邻 pair，直接结束。
- `best_pair` 应该每轮重新计算。
- `next_tokens` 应该每次应用 pair 前重新创建。
- 合并后要更新 `byte_tokens`。
- 最终 ids 应从最后的 `byte_tokens` 构造。

### 2.17 BPE training

Tokenizer inference 是“使用已有 vocab/merges 编码文本”；BPE training 是“从语料中学出 vocab/merges”。

`run_train_bpe(input_path, vocab_size, special_tokens)` 的输入输出：

```text
input_path: 训练文本文件路径
vocab_size: 最终词表最大大小，包含 special tokens、256 个 byte token、merge 产生的新 token
special_tokens: 加入 vocab 的特殊字符串，同时作为训练时的硬边界

返回:
vocab: dict[int, bytes]
merges: list[tuple[bytes, bytes]]
```

测试要求：

- `test_train_bpe`: merges 必须和 reference 完全一致；vocab 的 key 集合和值集合要一致。
- `test_train_bpe_speed`: 小语料 `corpus.en`、`vocab_size=500` 要在 1.5 秒内完成。
- `test_train_bpe_special_tokens`: special token 要加入 vocab，但训练统计时不能被普通 merge 吞进去；普通 vocab token 中不应包含 special token 片段。

训练流程：

```text
读文本
    -> 按 special tokens 切开，special token 不参与统计
    -> 对普通文本片段做 GPT-2 regex pretokenization
    -> 统计每个 pretoken 的频率
    -> 把每个 pretoken 转成 tuple[bytes, ...]，初始为单字节 token 序列
    -> 反复统计相邻 pair 的总频率
    -> 选择频率最高的 pair，平局时选 lexicographically greater 的 pair
    -> 记录这个 merge，并把语料中的该 pair 合并
    -> 新 token 加入 vocab
    -> 直到 vocab 达到 vocab_size
```

关键点：

- 初始 vocab 包含 256 个 byte token 和所有 special tokens。
- special tokens 是 hard boundary：它们把文档切开，不能跨它们统计 pair。
- pair 频率不是出现一次算一次，而是要乘以 pretoken 的频率。
- 平局规则很重要：同频 pair 选字典序更大的 pair。
- 先写清楚但可能较慢的版本，再优化 pair count 更新；不要一开始就写复杂并行。

当前建议实现顺序：

1. 实现 adapter，把 `run_train_bpe` 接到自己模块里的函数。
2. 写初始化 vocab：256 byte tokens + special tokens。
3. 写 special token split + pretokenization，并统计 pretoken 频率。
4. 写一轮 pair count 和 best pair 选择。
5. 写“把一个 best pair 应用到所有 pretoken token 序列”的逻辑。
6. 循环到 vocab size，并返回 vocab / merges。
7. 跑 `uv run pytest tests/test_train_bpe.py -q`，再根据速度优化。

实现笔记：读取训练文件

- `open(input_path, "r", encoding="utf-8")` 可以读取文本文件；`input_path` 可以是 `str`，也可以是 `PathLike`。
- `with open(...) as f:` 会在代码块结束时自动关闭文件。
- `f.read()` 会一次性读入整个文件，适合先在小 fixture 上把训练逻辑写通。
- `for line in f:` 是逐行读，内存更小，但训练 BPE 时还要考虑 special token 和 pretoken 边界。
- `f.read(chunk_size)` 可以固定大小分块读，适合后续大文件优化。
- 当前第一版实现建议先用 `f.read()` 完成清晰正确的串行版本；之后再用 chunking/multiprocessing 优化。

第十三版 train_bpe 代码审查：

- 初始 256 个 byte vocab 和 special token vocab 的方向正确。
- `re.finditer` 当前只收集了 special token 本身，没有收集 special token 之间的普通文本；训练应当把 special token 当作分隔边界，而不是把它们作为待统计 pretoken。
- 训练状态不能只有一条全局 `tokens` 序列；需要保存多个独立 pretoken 的 byte 序列及其频率，避免 pair 跨 pretoken 或 special token 边界。
- BPE 的训练循环应在完成全部 pretoken 统计后开始，而不是在遍历每个 pretoken 时开始。
- `len(tokens)` 是当前序列长度，不是 vocab 大小；停止条件应围绕当前 vocab 大小和是否存在可合并 pair。
- pair count 应表示 `pair -> 总频率`，并且每一轮重新计算或正确增量更新，不能把上一轮 count 无条件带入下一轮。
- 单个序列的 merge 需要左到右处理不重叠 pair，并保留最后一个未合并 token；之后要对所有 pretoken 序列应用该 merge。

第十四版 train_bpe 代码审查：

- 当前代码已经正确初始化 256 个 byte tokens，并尝试把 special tokens 加入 vocab。
- 当前 `pretokens` 实际保存了普通片段和 special token 匹配结果；special token 不应作为训练样本参与统计。
- special token 分割后还缺少最后一个普通文本尾段；没有 special token 时也必须把整个文本作为普通输入。
- 普通片段还没有经过 GPT-2 regex pretokenization，因此目前仍不是正确的 pretoken 集合。
- BPE merge 循环不能放在逐个 pretoken 的循环内部；必须先完成整个语料的 pretoken 统计，再进行全局 merge。
- 需要保存 `pretoken byte sequence -> frequency`，不能只有一条全局 `tokens` 序列。
- `len(tokens) > vocab_size` 不是训练停止条件，且 `max_pair` 在 while 条件中首次使用前没有初始化；停止条件应围绕 vocab 当前大小，并处理“没有可合并 pair”的情况。
- pair 计数应按所有 pretoken 的频率汇总，并在每轮重新计算或做正确的增量更新。

第十五版 train_bpe 代码审查：

- special token 分割后的普通片段收集方向已经改善；但如果没有 special tokens，需要把整个文本作为普通片段，否则后续没有输入。
- GPT-2 regex pretokenization 已加入，方向正确。
- `tokens` 在构造每个 pretoken 的 byte 序列时没有初始化；每个 pretoken 都应对应一条独立 byte-token 序列。
- 当前还没有统计相同 pretoken 的频率，导致后续 pair count 无法按 pretoken 出现次数加权。
- BPE 训练循环的条件仍然写反，并且 `max_pair` 在首次判断前未定义。
- `count` 应该每一轮重新统计，当前把 `count` 放在 while 外会让上一轮 pair 计数污染下一轮。
- `totaltokens` 不能在每一轮 while 里从原始 `pretokens` 重新追加，否则会重复累积旧序列；训练状态应是一份当前 pretoken 序列集合，每轮 merge 后更新它。
- `tokens = next_tokens` 只改变循环变量本身，不会更新 `totaltokens` 中保存的那条序列；需要从数据结构设计上确认“当前序列集合”如何被替换。
- merge 应用仍需处理不重叠 pair 和最后一个未合并 token。

第十六版 train_bpe 代码审查：

- 初始 `totaltokens` 已经移动到训练循环前，这是正确方向。
- while 内部仍然从原始 `pretokens` 重新编码并统计 pair；这会忽略上一轮 merge 后的 `totaltokens` 状态。pair 统计应基于当前 `totaltokens`。
- `max_count` 和 `max_pair` 仍然在逐 pretoken 循环里重置；它们应代表一整轮全语料的最佳 pair。
- `count` 仍然定义在 while 外；每轮统计前需要重新开始，否则旧 pair 频率会污染新一轮。
- 当前 merge 应用没有保留长度为 1 的序列，也没有正确处理最后一个未合并 token。
- 当前 merge 应用仍使用 `for range(len(tokens)-1)`，不能表达“匹配后跳两个位置，不匹配跳一个位置”的不重叠合并规则。
- 下一步重点：在 while 每轮中只遍历当前 `totaltokens` 做 pair count，不再使用原始 `pretokens`。

第十七版 train_bpe 测试记录：

- `tests/test_train_bpe.py::test_train_bpe` 已通过，说明小语料上的 merges/vocab correctness 基本成立。
- `test_train_bpe_speed` 失败，`test_train_bpe_special_tokens` 在大 fixture 上卡住，说明当前主要瓶颈是训练复杂度，而不是 BPE 规则本身。
- 当前 `totaltokens: list[list[bytes]]` 会把每一次 pretoken 出现都存一遍；大文件中重复 pretoken 很多，每轮 merge 都扫描全部实例，复杂度过高。
- 下一步优化方向：把训练状态改成 `pretoken byte-token sequence -> frequency`。相同 pretoken 只保存一次，统计 pair 时按 frequency 加权。
- 这个优化同时有助于 speed test 和 special token 大文件测试。

第十八版 train_bpe 测试记录：

- `totaltokens` 已改成 `dict[tuple[bytes, ...], int]`，即“当前 pretoken byte-token sequence -> frequency”。
- pair count 已按 pretoken frequency 加权。
- merge 后相同的新 token sequence 会累加 frequency，避免覆盖。
- `max_pair` 用 `(count[pair], pair)` 作为排序 key，满足“频率最大，平局取字典序更大 pair”。
- `uv run pytest tests/test_train_bpe.py -q` 结果：3 passed in 9.18s。
- Module 2 的 tokenizer inference 和 BPE training 测试均已通过；后续可选优化是进一步降低 runtime，但当前测试已满足。

本模块检查点：

- [x] 能解释 byte-level BPE 的整体流程。
- [x] 能区分训练 BPE merges 和使用已有 merges 编码文本。
- [x] 能解释 special token 的处理原则。
- [x] 能读懂 tokenizer 测试在验证什么行为。
- [x] 能实现并解释 BPE training 的 pair frequency、tie break、merge update。
- [x] 相关测试通过。

## Module 3: Data Batching

状态：Done

目标：把一维 token id 数据切成语言模型训练用的输入和标签。

需要理解的问题：

- dataset 为什么是一维 token id 数组？
- `x` 和 `y` 为什么长度相同？
- `y` 为什么是 `x` 右移一位的目标？
- batch shape 为什么是 `(batch_size, context_length)`？
- 为什么采样起点不能越界？

相关测试：

- `tests/test_data.py`

当前接口：

```text
dataset: npt.NDArray
batch_size: int
context_length: int
device: str
```

接口理解：

- `dataset` 是一维 NumPy token id 数组，不是原始字符串；长度记为 `N`。
- `batch_size` 是一次抽取多少条训练样本，对应输出的第 0 维。
- `context_length` 是每条样本包含多少个输入 token，对应输出的第 1 维。
- `device` 指定返回 PyTorch tensor 放在哪个设备，例如 `cpu`、`cuda` 或 `mps`。
- 输出应是 `x`、`y` 两个 shape 为 `(batch_size, context_length)` 的整数 tensor。
- 每行 `y` 都是对应 `x` 向右移动一位，抽样起点必须保证 `y` 不越界。

本模块检查点：

- [x] 能画出 token 序列到 `(x, y)` 的对应关系。
- [x] 能解释 batch 中每行样本如何采样。
- [x] 相关测试通过。

进度记录：

- `uv run pytest tests/test_data.py -q` 已通过。
- 已理解 `context_length` 是每条训练样本的输入 token 数。
- 已理解合法随机起点范围是 `[0, len(dataset) - context_length)`。
- 已理解 `x = dataset[start:start+C]`，`y = dataset[start+1:start+C+1]`。
- 已理解 NumPy array 可以通过 `np.stack` 后转成 torch tensor。
- 已解决 `devide` 拼写错误；正确关键字是 `device`。

## Module 4: 基础 NN 组件

状态：Done

目标：实现 Transformer 会用到的基础张量操作和神经网络组件。

组件：

- Linear
- Embedding
- SiLU
- Softmax
- Cross entropy
- RMSNorm
- SwiGLU

学习顺序：

1. Linear：理解矩阵乘法和权重 shape。
2. Embedding：理解 token id 查表。
3. SiLU：理解逐元素激活函数。
4. Softmax：理解概率归一化和数值稳定。
5. Cross entropy：理解从 logits 到平均 loss。
6. RMSNorm：理解按 hidden dimension 做归一化。
7. SwiGLU：理解 Transformer FFN 中的门控结构。

当前进度：

- 开始 Module 4。
- 第一小节先讲 Linear 和 Embedding，因为它们是 Transformer 输入和每层投影的基础。
- `uv run pytest tests/test_model.py::test_linear -q` 已通过。
- `uv run pytest tests/test_model.py::test_embedding -q` 已通过。
- `uv run pytest tests/test_model.py::test_silu_matches_pytorch -q` 已通过。
- `uv run pytest tests/test_nn_utils.py::test_softmax_matches_pytorch -q` 已通过。
- `uv run pytest tests/test_nn_utils.py -k cross_entropy -q` 已通过。
- `uv run pytest tests/test_model.py -k rmsnorm -q` 已通过。
- `uv run pytest tests/test_model.py -k swiglu -q` 已通过。

需要理解的问题：

- Linear 的权重 shape 和输入输出 shape 如何对应？
- Embedding 为什么本质上是查表？
- Softmax 为什么需要注意数值稳定性？
- Cross entropy 如何从 logits 和 target class 得到标量 loss？
- RMSNorm 和 LayerNorm 的差别是什么？
- SwiGLU 中三组权重分别承担什么角色？

相关测试：

- `tests/test_model.py`
- `tests/test_nn_utils.py`

本模块检查点：

- [x] 每个组件都能写出输入 shape 和输出 shape。
- [x] 能从 snapshot 测试理解误差容忍度。
- [x] 相关测试通过。

## Module 5: Attention / RoPE

状态：In progress

目标：理解 Transformer 的核心计算：attention 和位置编码。

顺序：

1. Scaled dot-product attention
2. Causal mask
3. Multi-head self-attention
4. RoPE
5. Multi-head self-attention with RoPE

需要理解的问题：

- Q、K、V 分别是什么？
- attention score 的 shape 是什么？
- 为什么要除以 `sqrt(d_k)`？
- mask 应该在 softmax 前还是后使用？
- multi-head 的 reshape / transpose 为什么容易出错？
- RoPE 为什么只作用在 Q 和 K 上？

相关测试：

- `tests/test_model.py::test_scaled_dot_product_attention`
- `tests/test_model.py::test_4d_scaled_dot_product_attention`
- `tests/test_model.py::test_multihead_self_attention`
- `tests/test_model.py::test_rope`
- `tests/test_model.py::test_multihead_self_attention_with_rope`

当前进度：

- 运行基础测试集合后，剩余失败为 Attention/RoPE、Transformer 组合和 `gradient_clipping` 的 `NotImplementedError`。
- 先从 scaled dot-product attention 开始，因为 MHA 和 RoPE attention 都依赖它。
- `uv run pytest tests/test_model.py::test_scaled_dot_product_attention -q` 已通过。
- `uv run pytest tests/test_model.py::test_4d_scaled_dot_product_attention -q` 已通过。
- `uv run pytest tests/test_model.py::test_multihead_self_attention -q` 已通过。

本模块检查点：

- [ ] 能写出 Q/K/V/score/output 的 shape 流程。
- [ ] 能解释 causal mask 的方向。
- [ ] 能解释 RoPE 的输入输出不改变 shape。
- [ ] 相关测试通过。

## Module 6: Transformer Block / Transformer LM

状态：Not started

目标：把前面的组件组合成完整语言模型。

需要理解的问题：

- pre-norm Transformer block 是什么？
- residual connection 在哪里？
- attention 子层和 FFN 子层如何连接？
- token embedding 如何进入模型？
- final RMSNorm 和 LM head 的作用是什么？
- 输出 logits 的 shape 为什么是 `(batch_size, sequence_length, vocab_size)`？

相关测试：

- `tests/test_model.py::test_transformer_block`
- `tests/test_model.py::test_transformer_lm`
- `tests/test_model.py::test_transformer_lm_truncated_input`

本模块检查点：

- [ ] 能画出一个 Transformer block 的数据流。
- [ ] 能解释 full LM 的 forward pass。
- [ ] 相关测试通过。

## Module 7: Optimizer / Scheduler / Checkpoint

状态：Not started

目标：理解训练流程中的优化器、学习率变化、梯度裁剪和保存恢复。

组件：

- AdamW
- Gradient clipping
- Cosine learning rate schedule with warmup
- Save/load checkpoint

需要理解的问题：

- Adam 和 AdamW 的区别是什么？
- gradient clipping 为什么按全局 L2 norm 裁剪？
- warmup 和 cosine decay 分别解决什么问题？
- checkpoint 需要保存 model、optimizer 和 iteration 的哪些状态？

相关测试：

- `tests/test_optimizer.py`
- `tests/test_serialization.py`

本模块检查点：

- [ ] 能解释 AdamW 的状态变量。
- [ ] 能解释学习率 schedule 的三个阶段。
- [ ] 能说明 checkpoint 恢复后为什么 optimizer 状态也重要。
- [ ] 相关测试通过。

## Module 8: 整体验证和提交

状态：Not started

目标：确认所有模块协同工作，准备提交。

最终检查：

- [ ] `uv run pytest -q` 通过。
- [ ] 没有把大数据、缓存、模型权重等无关文件加入提交。
- [ ] 理解 `make_submission.sh` 会排除哪些文件。
- [ ] 能生成提交 zip。

命令：

```sh
cd /Users/qianmo/assignment1-basics
uv run pytest -q
./make_submission.sh
```

## Appendix: Python 基础

### A.1 推导式

推导式是一种用一行表达“遍历并构造新容器”的语法。它常用于从已有 list、dict、set 等数据结构生成新的数据结构。

最常见的四类：

```text
list comprehension:      [...]
dict comprehension:      {...: ...}
set comprehension:       {...}
generator expression:    (...)
```

list 推导式：

```python
squares = [x * x for x in nums]
```

等价于：

```python
squares = []
for x in nums:
    squares.append(x * x)
```

dict 推导式：

```python
reverse_vocab = {v: k for k, v in vocab.items()}
```

等价于：

```python
reverse_vocab = {}
for k, v in vocab.items():
    reverse_vocab[v] = k
```

set 推导式：

```python
unique_lengths = {len(word) for word in words}
```

等价于：

```python
unique_lengths = set()
for word in words:
    unique_lengths.add(len(word))
```

带 `if` 的推导式：

```python
even_nums = [x for x in nums if x % 2 == 0]
```

等价于：

```python
even_nums = []
for x in nums:
    if x % 2 == 0:
        even_nums.append(x)
```

带解包的推导式：

```python
merge_priority = {pair: rank for rank, pair in enumerate(merges)}
```

如果 `merges` 中每个元素是 `(left, right)`，也可以写成：

```python
merge_priority = {(left, right): rank for rank, (left, right) in enumerate(merges)}
```

读推导式时，按这个顺序理解：

```text
先看 for 部分：从哪里遍历，每次变量叫什么
再看 if 部分：是否过滤
最后看开头：每次生成什么
```

例如：

```python
{v: k for k, v in vocab.items()}
```

阅读顺序：

```text
for k, v in vocab.items(): 遍历原 vocab 的 key-value
v: k: 新 dict 中用原 value 当 key，用原 key 当 value
```

推导式适合短小清晰的转换。如果逻辑有多层判断、很多临时变量或副作用，普通 `for` 循环更好读。
