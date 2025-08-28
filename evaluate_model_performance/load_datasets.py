# -*- coding: utf-8 -*-

"""
Functions for fetching similarity datasets (modified for local file access)
"""

import os
import numpy as np
import pandas as pd
from sklearn.utils import Bunch

# 设置数据集存储的基础目录（可通过环境变量配置）
DATA_BASE_DIR = os.environ.get('WORD_SIMILARITY_DATA', '/Users/chouyinghan/my_mathlib_project/Demo_in_Matrix/similarity_data')

def _get_local_path(filename, subdirectory=None):
    """构建本地文件路径"""
    if subdirectory:
        return os.path.join(DATA_BASE_DIR, subdirectory, filename)
    return os.path.join(DATA_BASE_DIR, filename)


def fetch_MEN(which="all", form="natural"):
    """从本地文件读取MEN数据集"""
    filenames = {
        "dev": 'datasets/MEN/MEN_dataset_lemma_form.dev',
        "test": 'datasets/MEN/MEN_dataset_lemma_form.test',
        "all": 'datasets/MEN/MEN_dataset_lemma_form_full'
    }
    filepath = _get_local_path(filenames[which])
    
    # 尝试不同分隔符
    try:
        data = pd.read_csv(filepath, header=None, sep=" ", names=['word1', 'word2', 'score'])
    except:
        try:
            data = pd.read_csv(filepath, header=None, sep="\t", names=['word1', 'word2', 'score'])
        except:
            data = pd.read_csv(filepath, header=None, delim_whitespace=True, 
                              names=['word1', 'word2', 'score'])
    
    # 处理词形还原形式
    if form == "natural":
        # 移除词性标记（如 "-n"）
        data['word1'] = data['word1'].apply(lambda x: x.split('-')[0] if isinstance(x, str) else x)
        data['word2'] = data['word2'].apply(lambda x: x.split('-')[0] if isinstance(x, str) else x)
    elif form != "lem":
        raise ValueError("Unrecognized form argument. Use 'natural' or 'lem'")
    
    # 确保所有分数是数值类型
    data['score'] = pd.to_numeric(data['score'], errors='coerce')
    
    # 打印样本以验证
    print(f"Loaded MEN data sample (form={form}):")
    print(f"  word1: '{data.iloc[0]['word1']}', word2: '{data.iloc[0]['word2']}', score: {data.iloc[0]['score']}")
    
    return Bunch(
        X=data[['word1', 'word2']].values.astype("object"), 
        y=data['score'].values.astype(float) / 5.0
    )


def fetch_WS353(which="all"):
    """从本地文件读取WS353数据集"""
    filenames = {
        "all": 'datasets/ws-353/wordsim353-english.txt',
        "relatedness": 'datasets/ws-353/wordsim353-english-rel.txt',
        "similarity": 'datasets/ws-353/wordsim353-english-sim.txt',
        "set1": 'datasets/wordsim353/set1.csv',
        "set2": 'datasets/wordsim353/set2.csv'
    }
    filepath = _get_local_path(filenames[which])
    data = pd.read_csv(filepath, header=0, sep="\t")
    
    X = data.values[:, 0:2]
    y = data.values[:, 2].astype(float)

    if data.values.shape[1] > 3:
        sd = np.std(data.values[:, 2:15].astype(float), axis=1).flatten()
        return Bunch(X=X.astype("object"), y=y, sd=sd)
    return Bunch(X=X.astype("object"), y=y)

def fetch_RG65():
    """
    Fetch Rubenstein and Goodenough dataset for testing attributional and
    relatedness similarity

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        dictionary-like object. Keys of interest:
        'X': matrix of 2 words per column,
        'y': vector with scores,
        'sd': vector of std of scores if available (for set1 and set2)

    References
    ----------
    Rubenstein, Goodenough, "Contextual correlates of synonymy", 1965

    Notes
    -----
    Scores were scaled by factor 10/4
    """
    filepath = _get_local_path('datasets/rg65.csv')
    
    # 正确读取分号分隔的数据
    data = pd.read_csv(filepath, header=None, sep=';', names=['word1', 'word2', 'score'])
    
    # 检查数据格式
    print(f"Loaded RG65 data sample:\n{data.head(3)}")
    print(f"Score range: {data['score'].min()} to {data['score'].max()}")
    
    return Bunch(X=data[['word1', 'word2']].values.astype("object"),
                 y=data['score'].values.astype(float) * 10.0 / 4.0)


def fetch_SimLex999(which='all'):
    """从本地文件读取SimLex999数据集"""
    filepath = _get_local_path('datasets/SimLex-999/SimLex-999.txt')
    data = pd.read_csv(filepath, sep="\t")
    
    X = data[['word1', 'word2']].values
    y = data['SimLex999'].values
    sd = data['SD(SimLex)'].values
    conc = data[['conc(w1)', 'conc(w2)', 'concQ']].values
    POS = data[['POS']].values
    assoc = data[['Assoc(USF)', 'SimAssoc333']].values
    
    if which == 'all':
        idx = np.arange(len(X))
    elif which == '333':
        idx = np.where(assoc[:, 1] == 1)[0]
    else:
        raise ValueError("Unrecognized subset")
    
    return Bunch(X=X[idx].astype("object"), y=y[idx], sd=sd[idx], 
                 conc=conc[idx], POS=POS[idx], assoc=assoc[idx])


def fetch_SimVerb3500(which='all'):
    filenames = {
        "all": 'datasets/simverb/data/SimVerb-3500.txt',
        "dev": 'datasets/simverb/data/SimVerb-500-dev.txt',
        "test": 'datasets/simverb/data/SimVerb-3000-test.txt'
    }
    filepath = _get_local_path(filenames[which], subdirectory='')
    
    # 使用制表符分隔，指定列名
    data = pd.read_csv(filepath, sep="\t", header=None, 
                      names=['word1', 'word2', 'pos', 'score', 'relation'])
    
    print(f"Loaded SimVerb3500 data sample:\n{data.head(3)}")
    print(f"Total verb pairs: {len(data)}")
    
    return Bunch(
        X=data[['word1', 'word2']].values.astype("object"), 
        y=data['score'].values.astype(float),
        pos=data['pos'].values,  # 添加词性信息
        relation=data['relation'].values  # 添加关系类型
    )



def fetch_SCWS():
    """从本地文件读取SCWS数据集（带上下文）"""
    filepath = _get_local_path('datasets/ehuang_sim_wcontext/SCWS/ratings.txt', subdirectory='')
    
    try:
        # 手动读取和处理文件
        data = []
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                # 跳过空行
                if not line.strip():
                    continue
                
                # 分割字段 - 使用制表符分割
                parts = line.strip().split('\t')
                
                # 验证基本字段数量（至少15个字段）
                if len(parts) < 15:
                    continue
                
                try:
                    # 提取固定位置的字段
                    # 根据实际数据格式：
                    # 字段0: 行号
                    # 字段1: 单词1
                    # 字段2: 单词1词性
                    # 字段3: 单词2
                    # 字段4: 单词2词性
                    # 字段5: 单词1上下文
                    # 字段6: 单词2上下文
                    # 字段7: 平均分
                    # 字段8-18: 10个评分
                    word1 = parts[1]
                    word2 = parts[3]
                    
                    # 平均分在第8个字段（索引7）
                    avg_score = float(parts[7])
                    
                    # 分数在第9-19个字段（索引8-18）
                    scores = list(map(float, parts[8:19]))
                    
                    # 提取上下文
                    word1_context = parts[5]
                    word2_context = parts[6]
                    
                    # 尝试提取句子（如果有）
                    sentence = ""
                    if len(parts) > 19:
                        sentence = " ".join(parts[19:])
                    
                    # 添加到数据集
                    data.append({
                        'word1': word1,
                        'word2': word2,
                        'word1_context': word1_context,
                        'word2_context': word2_context,
                        'sentence': sentence,
                        'avg_score': avg_score,
                        'scores': scores
                    })
                    
                except (IndexError, ValueError) as e:
                    # 打印有问题的行以便调试
                    print(f"⚠️ 解析行时出错: {line.strip()}")
                    print(f"错误信息: {str(e)}")
                    continue
        
        # 转换为DataFrame
        df = pd.DataFrame(data)
        if len(df) > 0:
            print(f"成功加载SCWS数据集，共 {len(df)} 个词对")
            print(f"数据样本:\n{df.iloc[0]}")
        else:
            print("⚠️ SCWS数据集加载完成，但没有有效数据")
        
        # 提取关键信息
        X = df[['word1', 'word2']].values.astype("object")
        mean_scores = df['avg_score'].values.astype(float)
        all_scores = np.array(df['scores'].tolist())
        sd_scores = np.std(all_scores, axis=1)
        
        # 同时返回上下文信息
        contexts = df[['word1_context', 'word2_context', 'sentence']].values
        
        return Bunch(
            X=X, 
            y=mean_scores,
            sd=sd_scores,
            contexts=contexts,
            all_scores=all_scores
        )
    
    except Exception as e:
        import traceback
        print(f"❌ 加载SCWS数据集失败: {str(e)}")
        traceback.print_exc()
        # 返回空数据集防止后续崩溃
        return Bunch(
            X=np.array([], dtype="object").reshape(0, 2),
            y=np.array([], dtype=float),
            sd=np.array([], dtype=float),
            contexts=np.array([], dtype="object").reshape(0, 3),
            all_scores=np.array([], dtype=float).reshape(0, 11)
        )