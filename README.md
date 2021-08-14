![](https://img.shields.io/badge/license-MIT-blue.svg) 
![](https://img.shields.io/badge/Python-3.6.12-blue.svg)
![](https://img.shields.io/badge/torch-1.7.0-brightgreen.svg)
![](https://img.shields.io/badge/transformers-4.4.1-brightgreen.svg)
![](https://img.shields.io/badge/scikitlearn-0.24.0-brightgreen.svg)
![](https://img.shields.io/badge/tqdm-4.49.0-brightgreen.svg)
![](https://img.shields.io/badge/jsonlines-2.0.0-brightgreen.svg)
![](https://img.shields.io/badge/loguru-0.5.3-brightgreen.svg)



# SimCSE-Chinese-Pytorch

SimCSE在中文上的复现，无监督 + 有监督

### 1. 背景

最近看了SimCSE这篇论文，便对论文做了pytorch版的复现和评测

- 论文：https://arxiv.org/pdf/2104.08821.pdf
- 官方：https://github.com/princeton-nlp/SimCSE

### 2. 文件

```shell
> datasets		数据集文件夹
   > cnsd-snli
   > STS-B
> pretrained_model	各种预训练模型文件夹
> saved_model		微调之后保存的模型文件夹
  data_preprocess.py	snli数据集的数据预处理
  simcse_sup.py		有监督训练
  simcse_unsup.py	无监督训练
```

### 3. 使用

需要将公开数据集和预训练模型放到指定目录下， 并检查在代码中的位置是否对应

```python
# 预训练模型目录
BERT = 'pretrained_model/bert_pytorch'
model_path = BERT 
# 微调后参数存放位置
SAVE_PATH = './saved_model/simcse_unsup.pt'
# 数据目录
SNIL_TRAIN = './datasets/cnsd-snli/train.txt'
STS_TRAIN = './datasets/STS-B/cnsd-sts-train.txt'
STS_DEV = './datasets/STS-B/cnsd-sts-dev.txt'
STS_TEST = './datasets/STS-B/cnsd-sts-test.txt'
```

数据预处理(需要先执行此文件)：

```shell
python data_preprocess.py
```

无监督训练

```shell
python simcse_unsup.py
```

有监督训练

```shell
python simcse_sup.py
```

### 4. 下载

数据集：

- CNSD：https://github.com/pluto-junzeng/CNSD

预训练模型：

- [BERT](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz)
- [BERT-wwm](https://drive.google.com/file/d/1AQitrjbvCWc51SYiLN-cJq4e0WiNN4KY/view)
- [BERT-wwm-ext](https://drive.google.com/file/d/1iNeYFhCBJWeUsIlnW_2K6SMwXkM4gLb_/view)
- [RoBERTa-wwm-ext](https://drive.google.com/file/d/1eHM3l4fMo6DsQYGmey7UZGiTmQquHw25/view)

### 5. 测评

测评指标为spearman相关系数

无监督：batch_size=64，lr=1e-5，droupout_rate=0.3，pooling=cls， 抽样10000样本

| 模型            | STS-B dev | STS-B test |
| :-------------- | --------- | ---------- |
| BERT            | 0.7308    | 0.6725     |
| BERT-wwm        | 0.7229    | 0.6628     |
| BERT-wwm-ext    | 0.7271    | 0.6669     |
| RoBERTa-wwm-ext | 0.7558    | 0.7141     |

有监督：batch_size=64，lr=1e-5，pooling=cls

| 模型            | STS-B dev | STS-B test | 收敛所需样本数 |
| :-------------- | --------- | ---------- | -------------- |
| BERT            | 0.8016    | 0.7624     | 23040          |
| BERT-wwm        | 0.8022    | 0.7572     | 16640          |
| BERT-wwm-ext    | 0.8081    | 0.7539     | 33280          |
| RoBERTa-wwm-ext | 0.8135    | 0.7763     | 38400          |

### 6. 参考

- https://arxiv.org/pdf/2104.08821.pdf
- 苏剑林. (Apr. 26, 2021). 《中文任务还是SOTA吗？我们给SimCSE补充了一些实验 》[Blog post]. Retrieved from https://kexue.fm/archives/8348
- https://github.com/zhengyanzhao1997/NLP-model/tree/main/model/model/Torch_model/SimCSE-Chinese









