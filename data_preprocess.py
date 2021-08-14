# -*- encoding: utf-8 -*-

"""snli数据预处理"""

import time

import jsonlines
from tqdm import tqdm


def timer(func):
    """ time-consuming decorator 
    """
    def wrapper(*args, **kwargs):
        ts = time.time()
        res = func(*args, **kwargs)
        te = time.time()
        print(f"function: `{func.__name__}` running time: {te - ts:.4f} secs")
        return res
    return wrapper


@timer
def snli_preprocess(src_path: str, dst_path:str) -> None:
    """处理原始的中文snli数据

    Args:
        src_path (str): 原始文件地址
        dst_path (str): 输出文件地址
    """
    # 组织数据
    all_data = {}
    with jsonlines.open(src_path, 'r') as reader:
        for line in tqdm(reader):
            sent1 = line.get('sentence1')
            sent2 = line.get('sentence2')
            label = line.get('gold_label')
            if not sent1:
                continue
            if sent1 not in all_data:
                all_data[sent1] = {}
            if label == 'entailment':
                all_data[sent1]['entailment'] = sent2                
            elif label == 'contradiction':
                all_data[sent1]['contradiction'] = sent2  
    # 筛选
    out_data = [
            {'origin': k, 'entailment': v.get('entailment'), 'contradiction': v.get('contradiction')} 
            for k, v in all_data.items() if v.get('entailment') and v.get('contradiction')
        ]
    # 写文件
    with jsonlines.open(dst_path, 'w') as writer:
        writer.write_all(out_data)
            
            
if __name__ == '__main__':
    
    dev_src, dev_dst = 'datasets/cnsd-snli/cnsd_snli_v1.0.dev.jsonl', 'datasets/cnsd-snli/dev.txt'
    test_src, test_dst = 'datasets/cnsd-snli/cnsd_snli_v1.0.test.jsonl', 'datasets/cnsd-snli/test.txt'
    train_src, train_dst = 'datasets/cnsd-snli/cnsd_snli_v1.0.train.jsonl', 'datasets/cnsd-snli/train.txt'
    
    snli_preprocess(train_src, train_dst)
    snli_preprocess(test_src, test_dst)
    snli_preprocess(dev_src, dev_dst)
 