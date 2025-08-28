#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MuRP模型评估工具 - 用于分析多关系双曲词嵌入模型
支持从自然语言定义中训练得到的MuRP模型评估与比较
"""

import numpy as np
import pandas as pd
import torch
import re
import nltk
from nltk.corpus import wordnet as wn
from scipy.stats import pearsonr, spearmanr
from sklearn.utils import Bunch
import warnings
import os
import sys
import json
import logging
from collections import defaultdict
import traceback
from tqdm import tqdm
import glob
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体支持
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
sns.set(font_scale=1.2)
sns.set_style("whitegrid")

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

# 确保下载WordNet数据
try:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except Exception as e:
    logger.warning(f"下载NLTK资源时出错: {e}")

# 引入数据加载函数
try:
    from load_datasets import fetch_SimVerb3500, fetch_MEN, fetch_SimLex999, fetch_SCWS, fetch_WS353, fetch_RG65
except ImportError:
    logger.warning("未找到load_datasets模块，使用回退实现。")
    # 提供简单的回退实现
    def fetch_SimVerb3500(split='all'):
        return Bunch(X=np.array([], dtype=object), y=np.array([]))
    def fetch_MEN(split='all'):
        return Bunch(X=np.array([], dtype=object), y=np.array([]))
    def fetch_SimLex999():
        return Bunch(X=np.array([], dtype=object), y=np.array([]))
    def fetch_SCWS():
        return Bunch(X=np.array([], dtype=object), y=np.array([]))
    def fetch_WS353():
        return Bunch(X=np.array([], dtype=object), y=np.array([]))
    def fetch_RG65():
        return Bunch(X=np.array([], dtype=object), y=np.array([]))

class MuRPEvaluator:
    """MuRP模型评估器 - 专门用于分析多关系双曲词嵌入模型"""
    
    def __init__(self, project_dir, device=None, output_dir=None):
        self.models = {}
        self.results = {}
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.wordnet_pos_map = {
            'n': '名词', 'v': '动词', 'a': '形容词', 'r': '副词', 
            's': '形容词卫星'
        }
        self.project_dir = project_dir  # 存储项目根目录
        self.output_dir = output_dir or os.path.join(project_dir, "evaluation_results")
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"使用设备: {self.device}")
        logger.info(f"项目目录: {self.project_dir}")
        logger.info(f"结果输出目录: {self.output_dir}")
        
        self.lemmatizer = nltk.stem.WordNetLemmatizer()
        self.synset_cache = {}
        self.embeddings_cache = {}
        self.word_coverage = defaultdict(lambda: defaultdict(int))  # 跟踪各模型的词汇覆盖率
    
    def word_to_synset_ids(self, word, pos=None):
        """将单词映射到可能的WordNet同义词集ID"""
        cache_key = f"{word}:{pos}"
        if cache_key in self.synset_cache:
            return self.synset_cache[cache_key]
        
        # 转换词性标签为WordNet格式
        wn_pos = None
        if pos and isinstance(pos, str):  # 确保pos是字符串
            if pos.startswith('n'):
                wn_pos = wn.NOUN
            elif pos.startswith('v'):
                wn_pos = wn.VERB
            elif pos.startswith('a') or pos.startswith('s'):
                wn_pos = wn.ADJ
            elif pos.startswith('r'):
                wn_pos = wn.ADV
        
        # 获取同义词集
        synsets = wn.synsets(word, pos=wn_pos)
        
        # 提取8位数字ID
        synset_ids = []
        for synset in synsets:
            # 从偏移量生成8位ID
            offset = str(synset.offset()).zfill(8)
            synset_ids.append(offset)
        
        # 缓存结果
        self.synset_cache[cache_key] = synset_ids
        return synset_ids
    
    def load_murp_model(self, model_path, data_dir, model_type=None):
        """加载multirelational-poincare仓库训练的MuRP模型（双曲或欧氏）"""
        try:
            # 从文件名推断模型类型和维度
            filename = os.path.basename(model_path)
            
            # 如果未指定模型类型，则自动检测
            if not model_type:
                if "poincare" in filename.lower() or "murp" in filename.lower():
                    model_type = "MuRP"
                elif "euclidean" in filename.lower() or "mure" in filename.lower():
                    model_type = "MuRE"
                else:
                    model_type = "Unknown"
                
            # 提取维度信息
            match = re.search(r'_(\d+)\.pth$', filename)
            if not match:
                match = re.search(r'dim_?(\d+)', filename)
            dimension = int(match.group(1)) if match else 0
            
            logger.info(f"加载{model_type}模型 (维度={dimension}) 从: {model_path}")
            
            # 添加项目源代码目录到系统路径
            source_dir = os.path.join(self.project_dir, "multirelational-poincare")
            if not os.path.exists(source_dir):
                # 尝试其他可能的路径
                source_dir = os.path.join(os.path.dirname(model_path), "..")
            
            if source_dir not in sys.path:
                sys.path.insert(0, source_dir)  # 添加到路径开头
            
            logger.info(f"添加到系统路径: {source_dir}")
            
            try:
                from load_data import Data
                if model_type == "MuRP":
                    from model import MuRP
                else:
                    from model import MuRE
                logger.info("成功导入模型模块")
            except ImportError as e:
                logger.error(f"导入模型模块时出错: {e}")
                logger.error(f"当前系统路径: {sys.path}")
                return None
            
            # 加载数据获取实体映射
            try:
                d = Data(data_dir)
                entity_list = d.entities
                entity_id_map = {entity: idx for idx, entity in enumerate(entity_list)}
                logger.info(f"从数据集中加载了{len(entity_list)}个实体")
            except Exception as e:
                logger.error(f"加载数据时出错: {e}")
                return None
            
            # 加载模型检查点
            try:
                if torch.cuda.is_available() and self.device == "cuda":
                    checkpoint = torch.load(model_path)
                else:
                    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
                
                # 创建模型实例
                if model_type == "MuRP":
                    # MuRP需要三个参数
                    model = MuRP(d, dimension, self.device)
                else:
                    # MuRE只需要两个参数
                    model = MuRE(d, dimension)
                
                # 加载模型状态
                model.load_state_dict(checkpoint)
                model.eval()
                model.to(self.device)
                
                # 提取实体嵌入
                with torch.no_grad():
                    # 根据模型类型获取嵌入
                    if model_type == "MuRP":
                        entity_embeddings = model.Eh.weight.data.cpu().numpy()
                    else:
                        entity_embeddings = model.E.weight.data.cpu().numpy()
                
                logger.info(f"提取的实体嵌入形状: {entity_embeddings.shape}")
            except Exception as e:
                logger.error(f"加载模型状态时出错: {e}")
                traceback.print_exc()
                return None
            
            # 创建嵌入缓存
            self.embeddings_cache = {
                entity: entity_embeddings[idx] 
                for entity, idx in entity_id_map.items()
            }
            
            return {
                "entity_embeddings": entity_embeddings,
                "entity_id_map": entity_id_map,
                "entity_list": entity_list,
                "type": model_type,
                "dimension": dimension,
                "embeddings_cache": self.embeddings_cache,
                "path": model_path
            }
        except Exception as e:
            logger.error(f"加载模型时出错: {e}")
            traceback.print_exc()
            return None
    
    def compute_poincare_distance(self, u, v, epsilon=1e-5):
        """计算Poincaré双曲距离"""
        try:
            # 转换为numpy数组
            u = np.asarray(u, dtype=np.float32)
            v = np.asarray(v, dtype=np.float32)
            
            # 计算欧几里得范数
            norm_u_sq = np.sum(u**2)
            norm_v_sq = np.sum(v**2)
            norm_diff_sq = np.sum((u - v)** 2)
            
            # 避免数值不稳定
            denominator = max((1 - norm_u_sq) * (1 - norm_v_sq), epsilon)
            
            # 计算双曲距离
            inner_expr = 1 + 2 * norm_diff_sq / denominator
            
            # 确保内部表达式大于1
            if inner_expr <= 1:
                inner_expr = 1 + epsilon
                
            return np.arccosh(inner_expr)
        except Exception as e:
            logger.warning(f"计算Poincaré距离时出错: {e}")
            return np.linalg.norm(u - v)
    
    def compute_euclidean_similarity(self, u, v):
        """计算欧氏嵌入的余弦相似度"""
        u = np.asarray(u, dtype=np.float32)
        v = np.asarray(v, dtype=np.float32)
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)
        
        if norm_u == 0 or norm_v == 0:
            return 0.0
        
        cosine_sim = np.dot(u, v) / (norm_u * norm_v)
        return (cosine_sim + 1) / 2  # 归一化到[0,1]范围
    
    def compute_similarity_batch(self, model_info, words1, words2, pos_list=None, dataset_name=None):
        """批量计算相似度（支持双曲和欧氏模型）"""
        entity_id_map = model_info["entity_id_map"]
        entity_list = model_info["entity_list"]
        embeddings = model_info["entity_embeddings"]
        embeddings_cache = model_info.get("embeddings_cache", {})
        model_type = model_info["type"]
        model_name = f"{model_type}-{model_info['dimension']}D"
        
        # 批量获取同义词集ID
        all_synsets1 = []
        all_synsets2 = []
        for i, (word1, word2) in enumerate(zip(words1, words2)):
            # 安全处理POS标签
            pos = None
            if pos_list is not None and i < len(pos_list):
                pos_item = pos_list[i]
                # 处理不同类型的POS标签
                if isinstance(pos_item, (list, tuple, np.ndarray)):
                    # 取第一个元素作为主要词性
                    pos = str(pos_item[0]) if len(pos_item) > 0 else None
                elif isinstance(pos_item, str):
                    pos = pos_item
                else:
                    pos = None
            
            all_synsets1.append(self.word_to_synset_ids(word1, pos))
            all_synsets2.append(self.word_to_synset_ids(word2, pos))
        
        # 批量查找嵌入
        similarities = np.full(len(words1), np.nan)
        valid_indices = []
        missing_words = []
        
        for i, (word1, word2, synsets1, synsets2) in enumerate(zip(words1, words2, all_synsets1, all_synsets2)):
            # 检查是否为空列表
            if not synsets1 or not synsets2:
                missing_words.append((word1, word2, "无同义词集"))
                continue
                
            # 查找有效的同义词集ID
            valid_synsets1 = [sid for sid in synsets1 if sid in entity_list]
            valid_synsets2 = [sid for sid in synsets2 if sid in entity_list]
            
            if not valid_synsets1 or not valid_synsets2:
                reason = ""
                if not valid_synsets1:
                    reason += f"{word1}无有效嵌入；"
                if not valid_synsets2:
                    reason += f"{word2}无有效嵌入"
                missing_words.append((word1, word2, reason))
                continue
                
            # 更新词汇覆盖率统计
            if dataset_name:
                self.word_coverage[dataset_name][word1] += 1
                self.word_coverage[dataset_name][word2] += 1
                
            # 尝试从缓存中获取嵌入
            emb1 = None
            for sid in valid_synsets1:
                if sid in embeddings_cache:
                    emb1 = embeddings_cache[sid]
                    break
                elif sid in entity_id_map:
                    idx = entity_id_map[sid]
                    emb = embeddings[idx]
                    embeddings_cache[sid] = emb
                    emb1 = emb
                    break
            
            emb2 = None
            for sid in valid_synsets2:
                if sid in embeddings_cache:
                    emb2 = embeddings_cache[sid]
                    break
                elif sid in entity_id_map:
                    idx = entity_id_map[sid]
                    emb = embeddings[idx]
                    embeddings_cache[sid] = emb
                    emb2 = emb
                    break
            
            if emb1 is not None and emb2 is not None:
                # 根据模型类型计算相似度
                if model_type == "MuRP":
                    # 双曲模型：使用Poincaré距离的倒数作为相似度
                    distance = self.compute_poincare_distance(emb1, emb2)
                    similarity = 1 / (1 + distance)
                else:
                    # 欧氏模型：使用余弦相似度
                    similarity = self.compute_euclidean_similarity(emb1, emb2)
                
                similarities[i] = similarity
                valid_indices.append(i)
        
        # 记录缺失的词对
        if missing_words and dataset_name:
            missing_file = os.path.join(self.output_dir, f"{dataset_name}_{model_name}_missing_pairs.txt")
            with open(missing_file, "w", encoding="utf-8") as f:
                f.write(f"在数据集 {dataset_name} 上模型 {model_name} 无法处理的词对 ({len(missing_words)}):\n")
                for w1, w2, reason in missing_words:
                    f.write(f"{w1} - {w2}: {reason}\n")
            logger.info(f"已记录无法处理的词对到 {missing_file}")
        
        return similarities, valid_indices
    
    def load_models(self, model_dir, data_dir, model_type_filter=None):
        """加载指定目录中的所有模型，可通过model_type_filter筛选模型类型"""
        logger.info(f"从目录加载模型: {model_dir}")
        
        # 查找所有模型文件
        model_files = glob.glob(os.path.join(model_dir, "*.pth"))
        logger.info(f"找到{len(model_files)}个模型文件")
        
        if not model_files:
            logger.error("未找到模型文件!")
            return
        
        # 加载每个模型
        for model_path in tqdm(model_files, desc="加载模型"):
            # 自动检测模型类型
            filename = os.path.basename(model_path).lower()
            model_type = None
            
            if "poincare" in filename or "murp" in filename:
                model_type = "MuRP"
            elif "euclidean" in filename or "mure" in filename:
                model_type = "MuRE"
            
            # 如果设置了筛选器且不匹配，则跳过
            if model_type_filter and model_type != model_type_filter:
                logger.debug(f"跳过模型 {model_path}，类型不匹配筛选条件")
                continue
            
            model_info = self.load_murp_model(model_path, data_dir, model_type)
            if model_info:
                # 生成唯一模型名称 (类型-维度)
                model_name = f"{model_info['type']}-{model_info['dimension']}D"
                # 如果存在相同类型和维度的模型，添加序号
                counter = 1
                original_name = model_name
                while model_name in self.models:
                    model_name = f"{original_name}-{counter}"
                    counter += 1
                self.models[model_name] = model_info
                logger.info(f"已加载模型: {model_name}")
            else:
                logger.error(f"加载模型失败: {model_path}")
    
    def evaluate_on_dataset(self, dataset_name, dataset_loader_func):
        """在特定数据集上评估模型"""
        logger.info(f"\n{'='*50}\n在 {dataset_name} 上评估\n{'='*50}")
        
        # 加载数据集
        try:
            data = dataset_loader_func()
            word_pairs = data.X
            human_scores = data.y
            
            # 确保human_scores是NumPy数组
            if not isinstance(human_scores, np.ndarray):
                human_scores = np.array(human_scores)
            
            # 确保human_scores是一维数组
            if human_scores.ndim > 1:
                human_scores = human_scores.flatten()
            
            # 提取词性信息（如果可用）
            pos_info = None
            if hasattr(data, 'pos') and data.pos is not None:
                # 确保pos_info是列表或数组
                if isinstance(data.pos, (list, tuple, np.ndarray)):
                    pos_info = data.pos
                    logger.info(f"加载了{len(pos_info)}个词对的词性标签")
                else:
                    logger.warning(f"词性标签类型异常: {type(data.pos)}，忽略词性信息。")
            
            logger.info(f"加载了{len(word_pairs)}个词对")
        except Exception as e:
            logger.error(f"加载数据集 {dataset_name} 时出错: {e}")
            traceback.print_exc()
            return
        
        dataset_results = {}
        
        for model_name, model_info in self.models.items():
            logger.info(f"在 {dataset_name} 上评估 {model_name}...")
            
            words1 = [str(pair[0]).strip() for pair in word_pairs]
            words2 = [str(pair[1]).strip() for pair in word_pairs]
            
            # 批量计算相似度
            model_scores, valid_indices = self.compute_similarity_batch(
                model_info, words1, words2, pos_info, dataset_name
            )
            
            # 确保有有效的索引
            if not valid_indices:
                logger.warning(f"模型 {model_name} 在数据集 {dataset_name} 上没有有效的预测结果")
                dataset_results[model_name] = {
                    "pearson": np.nan, "spearman": np.nan,
                    "coverage": 0.0, "n_pairs": 0,
                    "model_type": model_info["type"],
                    "dimension": model_info["dimension"]
                }
                continue
                
            valid_model_scores = model_scores[valid_indices]
            valid_human_scores = human_scores[valid_indices]
            
            # 确保human_scores是数值数组
            if not isinstance(valid_human_scores, np.ndarray) or valid_human_scores.dtype.kind not in 'iuf':
                try:
                    valid_human_scores = np.array(valid_human_scores, dtype=np.float32)
                except:
                    logger.error(f"人工评分类型无效: {type(valid_human_scores)}")
                    continue
            
            coverage = len(valid_indices) / len(word_pairs)
            
            # 计算覆盖率
            missing_count = len(word_pairs) - len(valid_indices)
            if missing_count > 0:
                logger.info(f"  缺失 {missing_count} 个词对 ({coverage:.2%} 覆盖率)")
            
            # 计算相关性指标
            if len(valid_model_scores) > 5:  # 至少需要5对有效数据
                try:
                    # 确保数组形状一致
                    if valid_model_scores.ndim > 1:
                        valid_model_scores = valid_model_scores.flatten()
                    if valid_human_scores.ndim > 1:
                        valid_human_scores = valid_human_scores.flatten()
                    
                    # 计算相关系数
                    pearson_corr, pearson_p = pearsonr(valid_model_scores, valid_human_scores)
                    spearman_corr, spearman_p = spearmanr(valid_model_scores, valid_human_scores)
                    
                    # 保存散点图
                    self.plot_correlation(
                        valid_human_scores, valid_model_scores,
                        dataset_name, model_name,
                        pearson_corr, spearman_corr
                    )
                    
                except Exception as e:
                    logger.error(f"计算相关性时出错: {e}")
                    traceback.print_exc()
                    pearson_corr = spearman_corr = np.nan
                    pearson_p = spearman_p = np.nan
                
                dataset_results[model_name] = {
                    "pearson": pearson_corr,
                    "pearson_p": pearson_p,
                    "spearman": spearman_corr,
                    "spearman_p": spearman_p,
                    "coverage": coverage,
                    "n_pairs": len(valid_model_scores),
                    "model_type": model_info["type"],
                    "dimension": model_info["dimension"]
                }
                logger.info(f"  Pearson: {pearson_corr:.4f} (p={pearson_p:.4f}), "
                           f"Spearman: {spearman_corr:.4f} (p={spearman_p:.4f}), "
                           f"覆盖率: {coverage:.2%}")
            else:
                logger.warning(f"  有效预测不足: {len(valid_model_scores)}/{len(word_pairs)}")
                dataset_results[model_name] = {
                    "pearson": np.nan, "spearman": np.nan,
                    "pearson_p": np.nan, "spearman_p": np.nan,
                    "coverage": coverage, "n_pairs": len(valid_model_scores),
                    "model_type": model_info["type"],
                    "dimension": model_info["dimension"]
                }
        
        self.results[dataset_name] = dataset_results
    
    def plot_correlation(self, human_scores, model_scores, dataset_name, model_name, pearson, spearman):
        """绘制人工评分与模型评分的相关性散点图"""
        try:
            plt.figure(figsize=(10, 8))
            sns.scatterplot(x=human_scores, y=model_scores, alpha=0.6)
            
            # 添加最佳拟合线
            sns.regplot(x=human_scores, y=model_scores, scatter=False, color='red')
            
            plt.title(f'{dataset_name} - {model_name}\n'
                      f'Pearson: {pearson:.4f}, Spearman: {spearman:.4f}')
            plt.xlabel('人工评分')
            plt.ylabel('模型预测相似度')
            plt.tight_layout()
            
            # 保存图像
            plot_dir = os.path.join(self.output_dir, "plots")
            os.makedirs(plot_dir, exist_ok=True)
            plot_path = os.path.join(plot_dir, f"{dataset_name}_{model_name}_correlation.png")
            plt.savefig(plot_path, dpi=300)
            plt.close()
            
            logger.info(f"相关性散点图已保存至: {plot_path}")
        except Exception as e:
            logger.warning(f"绘制相关性图时出错: {e}")
    
    def plot_dimension_analysis(self, results_df):
        """绘制不同维度对模型性能的影响"""
        try:
            # 创建维度分析图目录
            plot_dir = os.path.join(self.output_dir, "plots")
            os.makedirs(plot_dir, exist_ok=True)
            
            # 按模型类型和维度分组计算平均性能
            dim_analysis = results_df.groupby(['Model_Type', 'Dimension']).agg({
                'Pearson': 'mean',
                'Spearman': 'mean',
                'Coverage': 'mean'
            }).reset_index()
            
            # 绘制Pearson相关性
            plt.figure(figsize=(12, 6))
            sns.lineplot(data=dim_analysis, x='Dimension', y='Pearson', 
                         hue='Model_Type', marker='o', linewidth=2)
            plt.title('不同维度对Pearson相关性的影响')
            plt.xlabel('嵌入维度')
            plt.ylabel('平均Pearson相关性')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, 'dimension_vs_pearson.png'), dpi=300)
            plt.close()
            
            # 绘制Spearman相关性
            plt.figure(figsize=(12, 6))
            sns.lineplot(data=dim_analysis, x='Dimension', y='Spearman', 
                         hue='Model_Type', marker='o', linewidth=2)
            plt.title('不同维度对Spearman相关性的影响')
            plt.xlabel('嵌入维度')
            plt.ylabel('平均Spearman相关性')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, 'dimension_vs_spearman.png'), dpi=300)
            plt.close()
            
            # 绘制覆盖率
            plt.figure(figsize=(12, 6))
            sns.lineplot(data=dim_analysis, x='Dimension', y='Coverage', 
                         hue='Model_Type', marker='o', linewidth=2)
            plt.title('不同维度对覆盖率的影响')
            plt.xlabel('嵌入维度')
            plt.ylabel('平均覆盖率')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, 'dimension_vs_coverage.png'), dpi=300)
            plt.close()
            
            logger.info("维度分析图表已生成")
        except Exception as e:
            logger.warning(f"绘制维度分析图时出错: {e}")
    
    def plot_model_comparison(self, results_df):
        """绘制不同模型在各数据集上的性能比较"""
        try:
            plot_dir = os.path.join(self.output_dir, "plots")
            os.makedirs(plot_dir, exist_ok=True)
            
            # 对每个数据集创建比较图
            for dataset in results_df['Dataset'].unique():
                dataset_data = results_df[results_df['Dataset'] == dataset]
                
                plt.figure(figsize=(14, 8))
                sns.barplot(data=dataset_data, x='Model', y='Pearson', 
                           hue='Model_Type')
                plt.title(f'{dataset} 数据集上的模型性能比较 (Pearson相关性)')
                plt.xlabel('模型')
                plt.ylabel('Pearson相关性')
                plt.xticks(rotation=45, ha='right')
                plt.grid(True, axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, f'{dataset}_model_comparison.png'), dpi=300)
                plt.close()
            
            logger.info("模型比较图表已生成")
        except Exception as e:
            logger.warning(f"绘制模型比较图时出错: {e}")
    
    def run_evaluation(self, datasets, selected_datasets=None):
        """在指定数据集上运行评估"""
        logger.info("开始嵌入模型评估...")
        
        if len(self.models) == 0:
            logger.error("未加载任何模型，中止评估。")
            return
        
        # 如果指定了数据集子集，则只评估这些数据集
        if selected_datasets:
            datasets = {k: v for k, v in datasets.items() if k in selected_datasets}
            logger.info(f"只评估选定的数据集: {list(datasets.keys())}")
        
        # 评估每个数据集
        for dataset_name, loader_func in tqdm(datasets.items(), desc="评估数据集"):
            try:
                logger.info(f"处理数据集: {dataset_name}")
                self.evaluate_on_dataset(dataset_name, loader_func)
            except Exception as e:
                logger.error(f"评估 {dataset_name} 时出错: {e}")
                traceback.print_exc()
                continue
    
    def print_results_summary(self):
        """打印结果摘要"""
        if not self.results:
            logger.info("没有结果可显示!")
            return None
        
        print("\n" + "="*80)
        print("MuRP模型评估结果摘要")
        print("="*80)
        
        # 整理结果为DataFrame
        all_results = []
        for dataset_name, dataset_results in self.results.items():
            for model_name, metrics in dataset_results.items():
                all_results.append({
                    "Dataset": dataset_name,
                    "Model": model_name,
                    "Model_Type": metrics.get("model_type", "Unknown"),
                    "Dimension": metrics.get("dimension", 0),
                    "Pearson": metrics.get("pearson", np.nan),
                    "Pearson_p": metrics.get("pearson_p", np.nan),
                    "Spearman": metrics.get("spearman", np.nan),
                    "Spearman_p": metrics.get("spearman_p", np.nan),
                    "Coverage": metrics.get("coverage", 0),
                    "N_Pairs": metrics.get("n_pairs", 0)
                })
        
        results_df = pd.DataFrame(all_results)
        
        # 按数据集和模型类型显示
        for dataset in results_df['Dataset'].unique():
            dataset_df = results_df[results_df['Dataset'] == dataset]
            print(f"\n{dataset}:")
            print("-" * 120)
            print(f"{'模型':<20} {'类型':<8} {'维度':>6} {'Pearson':>10} {'p值':>8} {'Spearman':>10} {'p值':>8} {'覆盖率':>10} {'有效词对':>8}")
            print("-" * 120)
            
            # 按维度排序
            dataset_df = dataset_df.sort_values(by=['Model_Type', 'Dimension'])
            
            for _, row in dataset_df.iterrows():
                pearson_str = f"{row['Pearson']:.4f}" if not pd.isna(row['Pearson']) else "N/A"
                p_pearson_str = f"{row['Pearson_p']:.4f}" if not pd.isna(row['Pearson_p']) else "N/A"
                spearman_str = f"{row['Spearman']:.4f}" if not pd.isna(row['Spearman']) else "N/A"
                p_spearman_str = f"{row['Spearman_p']:.4f}" if not pd.isna(row['Spearman_p']) else "N/A"
                coverage_str = f"{row['Coverage']:.2%}"
                print(f"{row['Model']:<20} {row['Model_Type']:<8} {row['Dimension']:>6} {pearson_str:>10} {p_pearson_str:>8} {spearman_str:>10} {p_spearman_str:>8} {coverage_str:>10} {row['N_Pairs']:>8}")
                
        return results_df
    
    def save_results(self, filename="evaluation_results.csv"):
        """保存结果到CSV"""
        if not self.results:
            logger.info("没有结果可保存!")
            return None
        
        all_results = []
        for dataset_name, dataset_results in self.results.items():
            for model_name, metrics in dataset_results.items():
                all_results.append({
                    "Dataset": dataset_name,
                    "Model": model_name,
                    "Model_Type": metrics.get("model_type", "Unknown"),
                    "Dimension": metrics.get("dimension", 0),
                    "Pearson_Correlation": metrics.get("pearson", np.nan),
                    "Pearson_p_Value": metrics.get("pearson_p", np.nan),
                    "Spearman_Correlation": metrics.get("spearman", np.nan),
                    "Spearman_p_Value": metrics.get("spearman_p", np.nan),
                    "Coverage": metrics.get("coverage", 0),
                    "Valid_Pairs": metrics.get("n_pairs", 0)
                })
        
        results_df = pd.DataFrame(all_results)
        
        # 保存完整结果
        results_file = os.path.join(self.output_dir, filename)
        results_df.to_csv(results_file, index=False)
        logger.info(f"详细结果已保存至: {results_file}")
        
        # 生成并保存摘要统计
        summary = results_df.groupby(['Model_Type', 'Dimension']).agg({
            'Pearson_Correlation': ['mean', 'std', 'count'],
            'Spearman_Correlation': ['mean', 'std', 'count'],
            'Coverage': ['mean', 'std']
        }).round(4)
        
        summary_file = os.path.join(self.output_dir, "evaluation_summary.csv")
        summary.to_csv(summary_file)
        logger.info(f"评估摘要已保存至: {summary_file}")
        
        # 生成可视化图表
        self.plot_dimension_analysis(results_df)
        self.plot_model_comparison(results_df)
        
        return results_df

def main():
    """主函数：运行评估流程"""
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='MuRP模型评估工具 - 分析多关系双曲词嵌入模型')
    parser.add_argument('--project-dir', type=str, default="./", 
                      help='项目根目录')
    parser.add_argument('--model-dir', type=str, default="./models", 
                      help='模型文件所在目录')
    parser.add_argument('--data-dir', type=str, default="./data/WN18RR", 
                      help='数据集所在目录')
    parser.add_argument('--output-dir', type=str, default=None, 
                      help='结果输出目录')
    parser.add_argument('--device', type=str, default=None, 
                      help='使用的设备 (cpu 或 cuda)')
    parser.add_argument('--model-type', type=str, default=None, 
                      choices=['MuRP', 'MuRE', None], 
                      help='筛选模型类型，默认评估所有类型')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                      help='指定要评估的数据集，默认评估所有数据集')
    
    args = parser.parse_args()
    
    # 检查依赖
    missing_deps = []
    required = ["torch", "scipy", "sklearn", "pandas", "numpy", "nltk", "matplotlib", "seaborn"]
    for dep in required:
        try:
            __import__(dep)
        except ImportError:
            missing_deps.append(dep)
    
    if missing_deps:
        print(f"缺少依赖包: {missing_deps}")
        print("请使用以下命令安装: pip install " + " ".join(missing_deps))
        return None, None
    
    # 初始化评估器
    evaluator = MuRPEvaluator(
        project_dir=args.project_dir,
        device=args.device,
        output_dir=args.output_dir
    )
    
    # 解析模型和数据目录
    model_dir = args.model_dir
    if not os.path.isabs(model_dir):
        model_dir = os.path.join(args.project_dir, model_dir)
    
    data_dir = args.data_dir
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(args.project_dir, data_dir)
    
    # 检查模型目录是否存在
    if not os.path.exists(model_dir):
        print(f"模型目录不存在: {model_dir}")
        model_dir = input("请输入模型目录的完整路径: ")
        if not os.path.exists(model_dir):
            print("无效路径，程序退出。")
            return
    
    # 检查数据目录是否存在
    if not os.path.exists(data_dir):
        print(f"数据目录不存在: {data_dir}")
        # 尝试其他可能的路径
        possible_data_dir = os.path.join(args.project_dir, "multirelational-poincare/data/WN18RR/")
        if os.path.exists(possible_data_dir):
            data_dir = possible_data_dir
            print(f"使用替代数据目录: {data_dir}")
        else:
            data_dir = input("请输入数据目录的完整路径: ")
            if not os.path.exists(data_dir):
                print("无效路径，程序退出。")
                return
    
    # 批量加载模型
    evaluator.load_models(model_dir, data_dir, model_type_filter=args.model_type)
    
    # 检查是否有模型加载成功
    if not evaluator.models:
        logger.error("没有成功加载任何模型，中止评估。")
        return None, None
    
    # 定义要评估的数据集
    datasets = {
        "SimVerb3500": lambda: fetch_SimVerb3500('all'),
        "SimVerb3500-dev": lambda: fetch_SimVerb3500('dev'),
        "SimVerb3500-test": lambda: fetch_SimVerb3500('test'),
        "MEN": lambda: fetch_MEN("all"),
        "MEN-dev": lambda: fetch_MEN("dev"),
        "MEN-test": lambda: fetch_MEN("test"),
        "SimLex999": fetch_SimLex999,
        "SCWS": fetch_SCWS,
        "WS353": fetch_WS353,
        "RG65": fetch_RG65
    }
    
    # 运行评估
    evaluator.run_evaluation(datasets, selected_datasets=args.datasets)
    results_df = evaluator.print_results_summary()
    
    if results_df is not None:
        # 保存结果
        evaluator.save_results()
    else:
        logger.warning("没有结果可保存")
    
    return evaluator, results_df

if __name__ == "__main__":
    evaluator, results = main()
