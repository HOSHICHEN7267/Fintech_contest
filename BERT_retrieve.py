import re
import os
import json
import argparse
import numpy as np

from tqdm import tqdm
import jieba  # 用於中文文本分詞
from rank_bm25 import BM25Okapi  # 使用BM25演算法進行文件檢索

from sentence_transformers import SentenceTransformer, util

import torch
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F

def load_data(source_path):
    # 獲取資料夾中的檔案列表
    masked_file_ls = os.listdir(source_path)
    
    # 讀取每個 .txt 文件的文本內容，並以檔案名（去掉 .txt）作為鍵，文本內容作為值存入字典
    corpus_dict = {
        int(file.replace('.txt', '')): read_txt(os.path.join(source_path, file))
        for file in tqdm(masked_file_ls) if file.endswith('.txt')  # 確保只處理 .txt 文件
    }
    
    return corpus_dict

def read_txt(file_path):
    # 讀取 .txt 文件內容
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def BM25_retrieve(qs, source, corpus_dict):
    filtered_corpus = [corpus_dict[int(file)] for file in source]

    # [TODO] 可自行替換其他檢索方式，以提升效能

    tokenized_corpus = [
        list(jieba.cut_for_search(re.sub(r'[^\w\s]', '', doc))) for doc in filtered_corpus
    ]

    bm25 = BM25Okapi(tokenized_corpus)  # 使用BM25演算法建立檢索模型
    
    qs = re.sub(r'[^\w\s]', '', qs) # 移除標點符號

    tokenized_query = list(jieba.cut_for_search(qs))  # 將查詢語句進行分詞

    ans = bm25.get_top_n(tokenized_query, list(filtered_corpus), n=10)  # 根據查詢語句檢索，返回最相關的文檔，其中n為可調整項
    
    # print(ans)

    a = ans[0]
    # 找回與最佳匹配文本相對應的檔案名

    res = [key for key, value in corpus_dict.items() if value == a]
    return res[0]  # 回傳檔案名


# sentence transformer
def SBERT_retrieve(qs, source, corpus_dict):
    
    # 1. 加載預訓練的 SentenceTransformer 模型
    # model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')   #28%
    model = SentenceTransformer('moka-ai/m3e-base') #56%
    # model = SentenceTransformer('paraphrase-xlm-r-multilingual-v1') #18%
    # model = SentenceTransformer('distiluse-base-multilingual-cased-v2') #50%
    

    # 2. 計算查詢語句的嵌入
    query_embedding = model.encode(qs, convert_to_tensor=True)
    # tokenized_query = list(jieba.cut_for_search(qs))

    # 3. 過濾語料庫 
    filtered_corpus = [corpus_dict[int(file)] for file in source]

    # 4. 計算語料庫中文檔的嵌入
    corpus_embeddings = model.encode(filtered_corpus, convert_to_tensor=True)

    # 5. 計算查詢與每個文檔的相似度
    similarities = util.cos_sim(query_embedding, corpus_embeddings)[0]

    # 6. 找到相似度最高的文檔索引
    best_match_index = similarities.argmax().item()
    return source[best_match_index]  # 返回最相關的文檔標號

# transformer
def mean_pooling(model_output, attention_mask):
    # 計算平均池化嵌入
    token_embeddings = model_output[0]  # 第一個元素是所有 token 的嵌入
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def BERT_retrieve(qs, source, corpus_dict):
    # 1. 加載 BERT-base-chinese 模型和分詞器
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model = BertModel.from_pretrained('bert-base-chinese')

    # 2. 計算查詢語句的嵌入
    query_inputs = tokenizer(qs, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        query_output = model(**query_inputs)
    query_embedding = mean_pooling(query_output, query_inputs['attention_mask'])

    # 3. 過濾語料庫
    filtered_corpus = [corpus_dict[int(file)] for file in source]

    # 4. 計算語料庫中文檔的嵌入
    corpus_embeddings = []
    for doc in filtered_corpus:
        inputs = tokenizer(doc, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            output = model(**inputs)
        embedding = mean_pooling(output, inputs['attention_mask'])
        corpus_embeddings.append(embedding)

    # 將所有嵌入轉換為 tensor
    corpus_embeddings = torch.stack(corpus_embeddings)

    # 5. 計算查詢與每個文檔的餘弦相似度
    similarities = F.cosine_similarity(query_embedding, corpus_embeddings, dim=1)

    # 6. 找到相似度最高的文檔索引
    best_match_index = similarities.argmax().item()
    return source[best_match_index]  # 返回最相關的文檔標號

def lcs_length(str1, str2):
    m, n = len(str1), len(str2)
    # 建立一個 (m+1) x (n+1) 的矩陣，初始化為 0
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # 動態規劃計算 LCS 長度
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]

def LCS(qs, source, corpus_dict):
    # 過濾語料庫
    filtered_corpus = [corpus_dict[int(file)] for file in source]

    # 清洗查詢語句並分詞
    qs = re.sub(r'[^\w\s]', '', qs)  # 移除標點符號
    tokenized_query = list(jieba.cut_for_search(qs))  # 將查詢語句進行分詞
    query_str = ''.join(tokenized_query)  # 將分詞結果拼接成一個字符串

    # 計算每個文檔與查詢之間的 LCS 長度
    lcs_scores = []
    for doc in filtered_corpus:
        # 清洗文檔並分詞
        doc = re.sub(r'[^\w\s]', '', doc)  # 移除標點符號
        tokenized_doc = list(jieba.cut_for_search(doc))
        doc_str = ''.join(tokenized_doc)  # 將分詞結果拼接成一個字符串

        # 計算 LCS 長度並存入列表
        score = lcs_length(query_str, doc_str)
        lcs_scores.append(score)

    # 找到相似度最高的文檔索引
    best_match_index = lcs_scores.index(max(lcs_scores))
    best_match_document = filtered_corpus[best_match_index]

    # 找回與最佳匹配文本相對應的檔案名
    res = [key for key, value in corpus_dict.items() if value == best_match_document]
    return res[0]  # 回傳檔案名


def combined_retrieve(qs, source, corpus_dict, alpha=0.6, beta=0.4):
    # 預處理查詢
    qs = re.sub(r'[^\w\s]', '', qs)  # 移除標點符號
    tokenized_query = list(jieba.cut_for_search(qs))

    # 使用 BM25 檢索
    filtered_corpus = [corpus_dict[int(file)] for file in source]
    tokenized_corpus = [list(jieba.cut_for_search(re.sub(r'[^\w\s]', '', doc))) for doc in filtered_corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(tokenized_query)

    print("bm25: ", bm25_scores)
    bm25_scores = [max(0, score) for score in bm25_scores]
    bm25_scores = np.array(bm25_scores)

    # 使用 SBERT 計算語義相似度
    model = SentenceTransformer('moka-ai/m3e-base')
    query_embedding = model.encode(qs, convert_to_tensor=True)
    corpus_embeddings = model.encode(filtered_corpus, convert_to_tensor=True)
    sbert_similarities = util.cos_sim(query_embedding, corpus_embeddings)[0].cpu().numpy()
    sbert_similarities = np.array(sbert_similarities)

    print("sbert: ", sbert_similarities) 

    # 結合 BM25 和 SBERT 的分數
    combined_scores = alpha * bm25_scores + beta * sbert_similarities

    # 根據結合分數排序
    best_match_index = combined_scores.argmax()
    return source[best_match_index]  # 返回最相關的文檔標號

def check_bm25(qs, source, corpus_dict):
    filtered_corpus = [corpus_dict[int(file)] for file in source]
    # print("Filtered Corpus:", filtered_corpus)
    for i, doc in enumerate(filtered_corpus):
        if not doc.strip():  # 檢查空白文本
            print(f"Document {i} is empty or whitespace.")
        elif len(doc) < 5:  # 假設極短文檔長度小於 5
            print(f"Document {i} is unusually short:", doc)


if __name__ == "__main__":
    # 使用argparse解析命令列參數
    parser = argparse.ArgumentParser(description='Process some paths and files.')
    parser.add_argument('--question_path', type=str, required=True, help='讀取發布題目路徑')  # 問題文件的路徑
    parser.add_argument('--source_path', type=str, required=True, help='讀取參考資料路徑')  # 參考資料的路徑
    parser.add_argument('--output_path', type=str, required=True, help='輸出符合參賽格式的答案路徑')  # 答案輸出的路徑

    args = parser.parse_args()  # 解析參數

    answer_dict = {"answers": []}  # 初始化字典

    with open(args.question_path, 'rb') as f:
        qs_ref = json.load(f)  # 讀取問題檔案

    source_path_insurance = os.path.join(args.source_path, 'insurance')  # 設定參考資料路徑
    corpus_dict_insurance = load_data(source_path_insurance)

    source_path_finance = os.path.join(args.source_path, 'finance')  # 設定參考資料路徑
    corpus_dict_finance = load_data(source_path_finance)

    with open(os.path.join(args.source_path, 'faq/pid_map_content.json'), 'rb') as f_s:
        key_to_source_dict = json.load(f_s)  # 讀取參考資料文件
        key_to_source_dict = {int(key): value for key, value in key_to_source_dict.items()}

    for q_dict in qs_ref['questions']:
        # check_bm25(q_dict['query'], q_dict['source'], corpus_dict_finance)

        if q_dict['category'] == 'finance':
            # 進行檢索
            retrieved = combined_retrieve(q_dict['query'], q_dict['source'], corpus_dict_finance)
            # 將結果加入字典
            answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})

        elif q_dict['category'] == 'insurance':
            retrieved = combined_retrieve(q_dict['query'], q_dict['source'], corpus_dict_insurance)
            answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})

        elif q_dict['category'] == 'faq':
            corpus_dict_faq = {key: str(value) for key, value in key_to_source_dict.items() if key in q_dict['source']}
            retrieved = combined_retrieve(q_dict['query'], q_dict['source'], corpus_dict_faq)
            answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})

        else:
            raise ValueError("Something went wrong")  # 如果過程有問題，拋出錯誤

    # 將答案字典保存為json文件
    with open(args.output_path, 'w', encoding='utf8') as f:
        json.dump(answer_dict, f, ensure_ascii=False, indent=4)  # 儲存檔案，確保格式和非ASCII字符
