import os

# 设置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 下载预训练数据集
for i in range(0, 17):
    os.system(f'huggingface-cli download --repo-type dataset --resume-download Skywork/SkyPile-150B data/2020-40_zh_head_{str(i).zfill(4)}.jsonl --local-dir Skywork')

# 下载 SFT 数据集
os.system(f'huggingface-cli download --repo-type dataset --resume-download BelleGroup/train_3.5M_CN --local-dir BelleGroup')


# 两个数据集下载之后 都需要做一定的处理

# 1. SkyPile-150B 数据集
"""
import json
from tqdm import tqdm
import os


def split_text(text, chunk_size=512):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


paths = os.listdir('data')

with open('pretrain.jsonl', 'a', encoding='utf-8') as pretrain:
    for input_file in tqdm(paths, desc="Processing files"):  # 添加文件级别的进度条
        with open('data/' + input_file, 'r', encoding='utf-8') as f:
            data = f.readlines()
            for line in tqdm(data, desc=f"Processing lines in {input_file}", leave=False):  # 添加行级别的进度条
                line = json.loads(line)
                text = line['text']
                chunks = split_text(text)
                for chunk in chunks:
                    pretrain.write(json.dumps({'text': chunk}, ensure_ascii=False) + '\n')
"""
# 2. SFT 数据集
# 将数据集转换为常用的格式
"""
import json
from tqdm import tqdm

def convert_message(data):
    message = [
        {"role": "system", "content": "你是一个AI助手"},
    ]
    for item in data:
        if item['from'] == 'human':
            message.append({'role': 'user', 'content': item['value']})
        elif item['from'] == 'assistant':
            message.append({'role': 'assistant', 'content': item['value']})
    return message

with open('BelleGroup_sft.jsonl', 'a', encoding='utf-8') as sft:
    with open('train_3.5M_CN.json', 'r') as f:
        data = f.readlines()
        for item in tqdm(data, desc="Processing", unit="lines"):
            item = json.loads(item)
            message = convert_message(item['conversations'])
            sft.write(json.dumps(message, ensure_ascii=False) + '\n')
"""