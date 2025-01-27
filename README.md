# Tiny-LLM

<div align="center">
  <img src="./images/sft_sample.png" alt="Pretrain Dataset" style="width:80%;">
</div>

## 00 写在最前

> *其实我很久之前就像要动手使用 torch 实现一个小型的 LLM，但是碍于一直没有大片空闲的时间。趁着过年在家整好手头有一些算力资源，就动手尝试训练了一下。我会在下面简单记录我的实验过程，也会对代码做详细的介绍和注释。如有纰漏，还请见谅~*

模型大小为8.2千万参数的Tiny-LLM模型，在单卡A100上进行预训练和SFT训练。Pretrain阶段使用 [SkyWork 150B](https://huggingface.co/datasets/Skywork/SkyPile-150B) 数据中的 10B tokens 进行预训练，耗时 18 小时。SFT 阶段使用 [BelleGroup](https://huggingface.co/datasets/BelleGroup/train_3.5M_CN) 350万条多轮对话中文数据集在单卡A100 进行训练，耗时 5 小时。

> 注：模型同时在单卡 A6000 （训练参数设置完全相同）进行训练，Pretrain 阶段耗时 36 小时，SFT 阶段耗时 12 小时。

在 Pretrain 阶段我也产生过了使用不同规模的参数量的模型进行训练，如下图所示蓝色loss曲线为 2.1 千万参数的模型，橙色loss曲线为 8.2 千万参数的模型。在相同训练参数设置下，8.2 千万参数的模型在可以更快的达到loss收敛的拐点。我在这个点做过多次 step model 保存，当预训练达到拐点时模型开始学会说一些连贯的话，出现类似“涌现”的现象。但不管loss的拐点来的快还是慢，他们最终的loss都会收敛到差不多的程度。但模型的表现效果确实天差地别。所以在大模型 Pretrain 阶段 除了 loss 的拐点外，其实loss的收敛程度并不能代表模型的最终表现效果。

<div align="center">
  <img src="images/pretrain-faind.png" alt="Pretrain Dataset" style="width:50%;">
</div>

还有就是想向大家强烈推荐 [Swanlab](https://swanlab.cn/)，这是一个非常好用的实验管理平台，可以完全平替 wandb，最重要的是 Swablab 可以在手机上查看训练进度，无敌好用。

<div align="center">
  <img src="images/swanlab.png" alt="Pretrain Dataset" style="width:50%;">
</div>

> 注：Swanlab 是我在实验室实习时使用的实验管理平台，非常好用，推荐大家使用。

***总的来说，自己第一次动手写和训练大模型确实带来的成就感蛮足的。今天就先写到这里，后续我也会在这个repo的基础上做一些 tiny-visionLLM，tiny-MoELLM之类的，如果有大佬想一起搞，那求之不得哇~***


## Usage

1. 首先默认大家都是安装好 CUDA 的 Pytorch的，然后 `pip install -r requirements.txt` 安装依赖。
2. 下载数据集可以参考[`dataset_download.py`](dataset_download.py)文件。另外数据集也需要做一些处理，具体参考[`dataset_download.py`](dataset_download.py)文件。
3. 训练 Tokenizer，可以直接使用本仓库的`tokenizer_k`，词表大小是 6144，可以直接使用。当然也可以训练自己的 Tokenizer，具体参考[`train_tokenizer.py`](./train_tokenizer.py)文件。（后续我会上传我训练tokenzer的文件到网盘，大家也可以稍微等等下载）
4. 训练 Pretrain 阶段，可以参考[`pretrain.py`](./pretrain.py)文件。
5. 训练 SFT 阶段，可以参考[`sft_full.py`](./sft_full.py)文件。


## 01 Tokenizer

在自然语言处理 (NLP) 中，Tokenizer 是一种将文本分解为较小单位（称为 token）的工具。这些 token 可以是词、子词、字符，甚至是特定的符号。Tokenization 是 NLP 中的第一步，直接影响后续处理和分析的效果。不同类型的 tokenizer 适用于不同的应用场景，以下是几种常见的 tokenizer 及其特点。

**BPE**(Byte Pair Encoding)是一种基于统计方法，通过反复合并频率最高的字符或字符序列对来生成子词词典。这种方法的优点在于其简单和高效，能够有效地处理未知词和罕见词，同时保持较低的词典大小。BPE 的合并过程是自底向上的，逐步将频率最高的字符对合并成新的子词，直到达到预定的词典大小或不再有高频的字符对。

示例：
```
Input: "lower"
Output: ["low", "er"]

Input: "newest"
Output: ["new", "est"]
```

在这个例子中，单词“lower”被分割成子词“low”和“er”，而“newest”被分割成“new”和“est”。这种方法有效地处理了词干和词缀，保持了单词的基本语义结构。

那我们本次就是用 BPE Tokenizer 来进行 Tokenization。那首先就需要创建一些配置文件，来配置我们的 Tokenizer。包括 `tokenizer_config.json` 和 `special_tokens_map.json`，以及 chat_template，这个 chat_template 是用来配置我们的对话模板的。

```python
def create_tokenizer_config(save_dir: str) -> None:
    """创建完整的tokenizer配置文件"""
    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": True,
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "pad_token": "<|im_end|>",
        "unk_token": "<unk>",
        "model_max_length": 1000000000000000019884624838656,
        "clean_up_tokenization_spaces": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "chat_template": (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
            "{% elif message['role'] == 'user' %}"
            "<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
            "{% elif message['role'] == 'assistant' %}"
            "<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\n' }}"
            "{% endif %}"
        )
    }

    # 保存主配置文件
    with open(os.path.join(save_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    # 创建special_tokens_map.json
    special_tokens_map = {
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "unk_token": "<unk>",
        "pad_token": "<|im_end|>",
        "additional_special_tokens": ["<s>", "</s>"]
    }
    with open(os.path.join(save_dir, "special_tokens_map.json"), "w", encoding="utf-8") as f:
        json.dump(special_tokens_map, f, ensure_ascii=False, indent=4)
```

然后就可以开始开心的训练了，详细的代码可以查看本仓库目录下的 [`train_tokenizer.py`](./train_tokenizer.py) 文件。

## 02 Dataset

### Pretrain Dataset

Pretrain Dataset 其实很好理解，在模型的 Pretrain阶段主要是为了让模型学习到语言的一些基本规律，也就是知识学习阶段。模型在这个阶段需要学会如何利用前面的 `token` 来预测下一个 `token`。

```python
class SkyWorkPretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = 0
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = f.readlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        sample = json.loads(self.data[index])
        text = f"{self.tokenizer.bos_token}{sample['text']}"
        input_id = self.tokenizer(text).data['input_ids'][:self.max_length]
        text_len = len(input_id)
        # 没满最大长度的剩余部分
        padding_len = self.max_length - text_len
        input_id = input_id + [self.padding] * padding_len
        # 0表示不计算损失
        loss_mask = [1] * text_len + [0] * padding_len

        input_id = np.array(input_id)
        X = np.array(input_id[:-1]).astype(np.int64)
        Y = np.array(input_id[1:]).astype(np.int64)
        loss_mask = np.array(loss_mask[1:]).astype(np.int64)
        return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(loss_mask)
        
```

在以上代码可以看出，我们的 `Pretrain Dataset` 主要是将 `text` 通过 `tokenizer` 转换成 `input_id`，然后将 `input_id` 拆分成 `X` 和 `Y`，其中 `X` 为 `input_id` 的前 `n-1` 个元素，`Y` 为 `input_id` 的后 `n-1` 个元素。`loss_mask` 主要是用来标记哪些位置需要计算损失，哪些位置不需要计算损失。如果你不太能明白，可以看下面的示意图。

<div align="center">
  <img src="./images/pretrain-dataset.png" alt="Pretrain Dataset" style="width: 85%;">
</div>

图中的 Input ids 就是经过 tokenizer 转换后的 `input_id`，其中 `X` 就是 `input_id` 的前 `n-1` 个元素，`Y` 就是 `Input ids` 的后 `n-1` 个元素。`Loss Mask` 就是标记哪些位置需要计算损失，当然在 Pretrain 阶段是要对所有的 Y 都计算损失的。

### SFT Dataset

> 注：详细代码可以查看本仓库目录下的 [`dataset.py`](dataset.py) 文件。

SFT Dataset 其实是一个多轮对话数据集，我们的目标是让模型学会如何进行多轮对话。在这个阶段我们的输入是上一轮的对话内容，输出是当前轮的对话内容。

```python
class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = 0
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = f.readlines()

    def __len__(self):
        return len(self.data)

    def generate_loss_mask(self, input_ids):
        # 生成 loss mask, 0 表示不计算损失, 1 表示计算损失
        mask = [0] * len(input_ids)
        a_sequence = [3, 1074, 537, 500, 203]  # <|im_start|>assistant\n
        a_length = len(a_sequence)
        n = len(input_ids)
        i = 0
        
        while i <= n - a_length:
            # 检查当前位置是否匹配目标子序列
            match = True
            for k in range(a_length):
                if input_ids[i + k] != a_sequence[k]:
                    match = False
                    break
            if match:
                # 从子序列结束的位置开始查找第一个4
                j = None
                for idx in range(i + a_length, n):
                    if input_ids[idx] == 4:
                        j = idx
                        break
                if j is not None:
                    start = i + a_length
                    end = j  # 结束位置设为j（包含4）
                    # 标记区间为1（包括start到end）
                    if start <= end:
                        for pos in range(start, end + 1):
                            if pos < len(mask):
                                mask[pos] = 1
                # 跳过当前子序列，避免重叠匹配
                i += a_length
            else:
                i += 1
        return mask

    def __getitem__(self, index: int):
        sample = json.loads(self.data[index])
        text = self.tokenizer.apply_chat_template(sample, tokenize=False, add_generation_prompt=False)
        input_id = self.tokenizer(text).data['input_ids'][:self.max_length]
        text_len = len(input_id)
        # 没满最大长度的剩余部分
        padding_len = self.max_length - text_len
        input_id = input_id + [self.padding] * padding_len
        # 0表示不计算损失
        loss_mask = self.generate_loss_mask(input_id)

        input_id = np.array(input_id)
        X = np.array(input_id[:-1]).astype(np.int64)
        Y = np.array(input_id[1:]).astype(np.int64)
        loss_mask = np.array(loss_mask[1:]).astype(np.int64)
        return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(loss_mask)
```

在 SFT 阶段，我这里使用的是多轮对话数据集，所以就需要区分哪些位置需要计算损失，哪些位置不需要计算损失。在上面的代码中，我使用了一个 `generate_loss_mask` 函数来生成 `loss_mask`。这个函数主要是用来生成 `loss_mask`，其中 `loss_mask` 的生成规则是：当遇到 `|<im_start|>assistant\n` 时，就开始计算损失，直到遇到 `|<im_end|>` 为止。这样就可以保证我们的模型在 SFT 阶段只计算当前轮的对话内容。那我也给出一个示意图，帮助大家理解。

<div align="center">
  <img src="./images/sft-dataset.png" alt="Pretrain Dataset" style="width: 85%;">
</div>

可以看到，其实 SFT Dataset 和 Pretrain Dataset 的 `X` 和 `Y` 是一样的，只是在 SFT Dataset 中我们需要生成一个 `loss_mask` 来标记哪些位置需要计算损失，哪些位置不需要计算损失。 图中 `Input ids` 中的蓝色小方格就是AI的回答，所以是需要模型学习的地方。所以在 `loss_mask` 中，蓝色小方格对应的位置是黄色，其他位置是灰色。在代码 `loss_mask` 中的 1 对应的位置计算损失，0 对应的位置不计算损失。

## 03 Model

首先是 [ModelConfig](k_model.py)，这个如果后续想要导出为 transformers 可以加载的模型，就需要定义一个 `ModelConfig` 类，且需要继承 `transformers.PretrainedConfig`。

```python
class ModelConfig(PretrainedConfig):
    model_type = "Tiny-K"
    def __init__(
            self,
            dim: int = 768,
            n_layers: int = 12,
            n_heads: int = 16,
            n_kv_heads: int = 8,
            vocab_size: int = 6144,
            hidden_dim: int = None,
            multiple_of: int = 64,
            norm_eps: float = 1e-5,
            max_seq_len: int = 512,
            dropout: float = 0.0,
            flash_attn: bool = True,
            **kwargs,
    ):
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.flash_attn = flash_attn
        super().__init__(**kwargs)
```

模型部分不会过多赘述，代码里面已经做了很详细的注释。这里只是简单的介绍一下模型的结构。想要详细了解模型的结构，可以查看本仓库目录下的 [`k_model.py`](k_model.py) 文件。

## 04 Training

终于到训练环节啦！

首先此次 Pretrain 阶段使用了余弦退火学习率调度器，代码如下：

> 注：代码参考自 llama2.c 仓库。

```python
def get_lr(it, all):
    """
    根据当前的训练迭代步数 it 返回当前的学习率值。
    学习率调整策略包括线性预热、余弦退火和最小学习率限制。
    """
    warmup_iters = args.warmup_iters
    lr_decay_iters = all
    min_lr = args.learning_rate / 10

    # 1) 线性预热阶段，在 warmup_iters 之前，学习率线性增加到目标学习率
    if it < warmup_iters:
        return args.learning_rate * it / warmup_iters
    
    # 2) 如果迭代步数超过 lr_decay_iters，返回最小学习率 min_lr
    if it > lr_decay_iters:
        return min_lr
    
    # 3) 余弦退火阶段，在 warmup_iters 和 lr_decay_iters 之间，学习率逐渐降低
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1 # 确保衰减比在合法范围内
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # 余弦函数计算衰减系数，范围为0到1
    return min_lr + coeff * (args.learning_rate - min_lr) # 根据衰减系数调整学习率
```

我们可以来看一下学习率的整体趋势，如下图所示：

<div align="center">
  <img src="./images/lr.png" alt="Pretrain Dataset" style="width: 85%;">
</div>

可以看到，学习率本应该在预热阶段是线性增加的，但我设置的 `warmup_iters=0`，哈哈哈。然后在余弦退火阶段逐渐降低，最后到达最小学习率。

OK，训练代码由于我只有单卡，所以也没有写 DDP 多卡并行（我真的很想尝试一下，有没有大佬施舍一点~）。Pretrain 和 SFT Train 的训练代码基本一样，只是 Dataset 形式不同，所以我就只展示 Pretrain 阶段的训练代码。

> 注：详细代码可以查看本仓库目录下的 [`pretrain.py`](./pretrain.py) 和 [`sft_full`](sft_full.py) 文件。

```python
def train_epoch(epoch):
    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with ctx:
            out = model(X, Y)
            loss = out.last_loss / args.accumulation_steps
            loss_mask = loss_mask.view(-1)
            loss = torch.sum(loss * loss_mask) / loss_mask.sum()

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.7f} epoch_Time:{}min:'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))
            if args.use_swanlab:
                swanlab.log({
                    "loss": loss.item() * args.accumulation_steps,
                    "lr": optimizer.param_groups[-1]['lr']
                })

        if (step + 1) % args.save_interval == 0:
            model.eval()
            ckp = f'{args.save_dir}/SkyWork_pretrain_{lm_config.dim}_{lm_config.n_layers}_{lm_config.vocab_size}.pth'

            state_dict = model.state_dict()
            torch.save(state_dict, ckp)
            model.train()
        
        if (step + 1) % 20000 == 0:
            model.eval()
            ckp = f'{args.save_dir}/SkyWork_pretrain_{lm_config.dim}_{lm_config.n_layers}_{lm_config.vocab_size}_step{step+1}.pth'

            state_dict = model.state_dict()
            torch.save(state_dict, ckp)
            model.train()
```

> 注：此处参考 minimind 仓库的代码。


## 参考链接

- [SkyWork 150B](https://huggingface.co/datasets/Skywork/SkyPile-150B)
- [BelleGroup](https://huggingface.co/datasets/BelleGroup/train_3.5M_CN)
- [llama2.c](https://github.com/karpathy/llama2.c)
- [minimind](https://github.com/jingyaogong/minimind)