import json
import numpy as np
import argparse
import os

def compare_json_files(ground_truth_path, prediction_path):
    # 1. 讀取 JSON 文件

    with open(ground_truth_path, 'rb') as f:
        ground_truth_data = json.load(f)  # 讀取問題檔案

    with open(prediction_path, 'r', encoding='utf-8') as file:
        prediction_data = json.load(file)
    
    # print(ground_truth_data)


    # 2. 建立字典以便快速查找 ground truth 的 retrieve 值
    # ground_truth_dict = {item['qid']: item['retrieve'] for item in ground_truth_data['ground_truths']}
    ground_truth_dict = {}
    for item in ground_truth_data['ground_truths']:
        # print(item['qid'])
        qid = item['qid']
        retrieve = item['retrieve']
        if qid not in ground_truth_dict:
            ground_truth_dict[qid] = []
        ground_truth_dict[qid].append(retrieve)

    # print(ground_truth_dict)


    # 3. 初始化變量以計算正確率
    total = 0
    correct = 0

    # print(prediction_data['answers'])
    # 4. 比對預測結果
    for item in prediction_data['answers']:
        qid = item['qid']
        predicted_retrieve = item['retrieve']
        
        if qid in ground_truth_dict:
            total += 1
            if int(ground_truth_dict[qid][0]) == int(predicted_retrieve):
                correct += 1

    # 5. 計算正確率
    accuracy = correct / total if total > 0 else 0
    print(f"正確率：{correct}/{total} = {accuracy:.2f}")


subject = 'sbert_dis.json'

ground_truth_path = './QA/QA_organized/testANSFinance.json'
prediction_path = './QA/answer/Finance/'
print("Finance:", end = " " )
compare_json_files(ground_truth_path, os.path.join(prediction_path, subject))

ground_truth_path = './QA/QA_organized/testANSInsurance.json'
prediction_path = './QA/answer/Insurance/'
print("Insurance:", end = " " )
compare_json_files(ground_truth_path, os.path.join(prediction_path, subject))

ground_truth_path = './QA/QA_organized/testANSFAQ.json'
prediction_path = './QA/answer/FAQ/'
print("FAQ:", end = " " )
compare_json_files(ground_truth_path, os.path.join(prediction_path, subject))