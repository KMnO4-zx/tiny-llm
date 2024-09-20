import os
import struct
from sentencepiece import SentencePieceProcessor
from typing import List

TOKENIZER_MODEL = "tokenizer.model"

class Tokenizer:
    def __init__(self, tokenizer_model=None):
        """
        初始化分词器。加载预训练的SentencePiece模型，并设置一些特殊的token ID。

        参数:
        tokenizer_model: str, 可选，分词器模型的路径，如果不指定则使用默认路径 TOKENIZER_MODEL。
        """
        # 如果提供了分词器模型路径，使用该路径；否则使用默认模型路径
        model_path = tokenizer_model if tokenizer_model else TOKENIZER_MODEL
        # 确保模型文件存在
        assert os.path.isfile(model_path), model_path

        # 加载 SentencePiece 模型
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        self.model_path = model_path

        # 获取分词器的特殊token和词汇表大小
        self.n_words: int = self.sp_model.vocab_size()  # 词汇表大小
        self.bos_id: int = self.sp_model.bos_id()       # 句子开头 (BOS) 的ID
        self.eos_id: int = self.sp_model.eos_id()       # 句子结尾 (EOS) 的ID
        self.pad_id: int = self.sp_model.pad_id()       # 填充 (PAD) 的ID

        # 验证分词器词汇表大小是否正确
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        """
        将字符串编码为词元ID列表。可以选择是否添加句子开头 (BOS) 和句子结尾 (EOS) 标记。

        参数:
        s: str, 要编码的字符串。
        bos: bool, 是否在编码的词元列表前添加 BOS 标记。
        eos: bool, 是否在编码的词元列表末尾添加 EOS 标记。

        返回:
        List[int]: 编码后的词元ID列表。
        """
        # 确保输入是字符串类型
        assert type(s) is str
        # 使用SentencePiece将字符串编码为词元ID
        t = self.sp_model.encode(s)
        # 如果需要BOS标记，将其添加到词元列表开头
        if bos:
            t = [self.bos_id] + t
        # 如果需要EOS标记，将其添加到词元列表末尾
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        """
        将词元ID列表解码为字符串。

        参数:
        t: List[int], 词元ID列表。

        返回:
        str: 解码后的字符串。
        """
        return self.sp_model.decode(t)

    def export(self):
        """
        将分词器的词元和对应的得分导出为二进制文件，便于后续高效加载使用。
        """
        # 存储所有的词元和它们的得分
        tokens, scores = [], []
        for i in range(self.n_words):
            # 根据ID获取词元并进行轻量化后处理
            t = self.sp_model.id_to_piece(i)  # 获取词元
            s = self.sp_model.get_score(i)    # 获取词元的得分
            # 如果是BOS或EOS标记，替换为可读的格式
            if i == self.bos_id:
                t = '\n<s>\n'
            elif i == self.eos_id:
                t = '\n</s>\n'
            # SentencePiece使用的'▁'字符代表空格，这里替换回空格
            t = t.replace('▁', ' ') 
            # 将词元转换为UTF-8编码的字节格式
            b = t.encode('utf-8')

            # 将词元和对应的得分存入列表
            tokens.append(b)
            scores.append(s)

        # 计算所有词元中的最大长度（以字节数计算）
        max_token_length = max(len(t) for t in tokens)

        # 将分词器的词元和得分写入二进制文件
        tokenizer_bin = self.model_path.replace('.model', '.bin')
        with open(tokenizer_bin, 'wb') as f:
            # 先写入最大词元长度
            f.write(struct.pack("I", max_token_length))
            # 依次写入每个词元的得分、长度和字节数据
            for bytes, score in zip(tokens, scores):
                f.write(struct.pack("fI", score, len(bytes)))  # 写入得分和字节长度
                f.write(bytes)  # 写入词元的字节数据
