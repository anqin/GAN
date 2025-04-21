#!/usr/bin/env python3
# encoding: utf-8
# coding style: pep8
# ====================================================
#   Copyright (C) 2025 ANQIN-X Project. All rights reserved.
#
#   Author        : An Qin
#   Email         : anqin.qin@gmail.com
#   File Name     : test_torch.py
#   Last Modified : 2025-04-21 15:43
#   Describe      : 
#
# ====================================================

import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer


# 基于deepseek r1为基础进行调训的模型类（7B）
class FineTunedDeepSeek7BModel(torch.nn.Module):
    def __init__(self):
        super(FineTunedDeepSeek7BModel, self).__init__()
        self.model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-llm-7b-r1")
        self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-llm-7b-r1")

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)


# 基于deepseek 1.5b r1为基础进行调训的模型类
class FineTunedDeepSeek1_5BModel(torch.nn.Module):
    def __init__(self):
        super(FineTunedDeepSeek1_5BModel, self).__init__()
        self.model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-llm-1.5b-r1")
        self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-llm-1.5b-r1")

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)


# 基于qwen模型进行调训的模型类
class FineTunedQwenModel(torch.nn.Module):
    def __init__(self):
        super(FineTunedQwenModel, self).__init__()
        self.model = AutoModelForCausalLM.from_pretrained("/home/anqin/download/qwen")
        self.tokenizer = AutoTokenizer.from_pretrained("/home/anqin/download/qwen")

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)


def train_model(problem_data_list, model_choice="deepseek_7b"):
    if model_choice == "deepseek_7b":
        model = FineTunedDeepSeek7BModel()
    elif model_choice == "deepseek_1_5b":
        model = FineTunedDeepSeek1_5BModel()
    else:
        model = FineTunedQwenModel()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    start_time = time.time()
    for epoch in range(2):  # 这里减少训练轮数以加快示例速度，实际应用中可调整
        print(f"开始第 {epoch + 1} 轮训练...")
        for idx, problem_data in enumerate(problem_data_list):
            print(f"正在处理第 {idx + 1} 个数据样本...")
            problem = problem_data["problem"]
            correct_agents_thinking = problem_data["correct_agents_thinking"]
            if not correct_agents_thinking:
                continue

            # 准备输入和目标数据
            input_text = problem + " ".join(correct_agents_thinking)
            inputs = model.tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            labels = input_ids.clone()

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            print(f"第 {idx + 1} 个数据样本处理完毕，当前损失值: {loss.item()}")
        print(f"第 {epoch + 1} 轮训练结束")
    end_time = time.time()
    print(f"整个训练过程耗时: {end_time - start_time} 秒")
    return model


def main():
    # 构造3-4个竞赛水平的数据难题测试数据
    test_problem_data_list = [
        {
            "problem": "设正整数 \( n \geq 2 \)，且 \( a_1,a_2,\cdots,a_n \) 是 \( n \) 个互不相同的正整数，满足 \( \sum_{i = 1}^{n}a_i=2024 \)。求 \( \sum_{1\leq i\lt j\leq n}a_ia_j \) 的最大值。",
            "correct_agents_thinking": ["首先，根据\((\sum_{i = 1}^{n}a_i)^2=\sum_{i = 1}^{n}a_{i}^{2}+2\sum_{1\leq i\lt j\leq n}a_ia_j\)，则\(\sum_{1\leq i\lt j\leq n}a_ia_j=\frac{(\sum_{i = 1}^{n}a_i)^2 - \sum_{i = 1}^{n}a_{i}^{2}}{2}\)。要使\(\sum_{1\leq i\lt j\leq n}a_ia_j\)最大，需使\(\sum_{i = 1}^{n}a_{i}^{2}\)最小。因为\(a_1,a_2,\cdots,a_n\)是互不相同的正整数，由排序不等式可知，当\(a_i = i\)（\(i = 1,2,\cdots,n\)）时，\(\sum_{i = 1}^{n}a_{i}^{2}\)取得最小值。通过计算\(\sum_{i = 1}^{n}i=\frac{n(n + 1)}{2}\)，找到满足\(\frac{n(n + 1)}{2}\leq2024\)的最大\(n\)，\(n = 63\)时，\(\sum_{i = 1}^{63}i=\frac{63\times64}{2}=2016\)，则\(a_1 = 1,a_2 = 2,\cdots,a_{62}=62,a_{63}=76\) ，再代入\(\sum_{1\leq i\lt j\leq n}a_ia_j=\frac{(\sum_{i = 1}^{n}a_i)^2 - \sum_{i = 1}^{n}a_{i}^{2}}{2}\)计算出最大值。"]
        },
        {
            "problem": "在平面直角坐标系 \( xOy \) 中，已知椭圆 \( C:\frac{x^{2}}{a^{2}}+\frac{y^{2}}{b^{2}} = 1(a>b>0) \) 的离心率为\(\frac{\sqrt{3}}{2}\)，且过点\((2,\sqrt{2})\)。设\(A\)，\(B\)分别为椭圆\(C\)的左、右顶点，\(P\)为椭圆\(C\)上异于\(A\)，\(B\)的动点，直线\(AP\)，\(BP\)分别与直线\(x = 8\)相交于\(M\)，\(N\)两点，以线段\(MN\)为直径的圆是否恒过\(x\)轴上的定点？若过定点，求出定点坐标；若不过定点，请说明理由。",
            "correct_agents_thinking": ["先根据椭圆的离心率和过的点求出椭圆方程，设\(P(x_0,y_0)\)，根据点\(P\)在椭圆上得到\(y_0\)与\(x_0\)的关系。再根据\(A\)，\(B\)坐标求出直线\(AP\)，\(BP\)方程，进而得到\(M\)，\(N\)坐标。设以\(MN\)为直径的圆过\(x\)轴上的定点\(Q(x_1,0)\)，根据\(\overrightarrow{MQ}\cdot\overrightarrow{NQ}=0\)列出关于\(x_0\)，\(x_1\)的方程，通过化简方程，求解出\(x_1\)的值，从而确定定点坐标。"]
        },
        {
            "problem": "有一个数列\(\{a_n\}\)满足\(a_1 = 1\)，\(a_{n + 1}=\frac{a_n}{1 + 3a_n}\)，\(n\in N^*\)。若\(b_n=\frac{1}{a_n}+2^n\)，求数列\(\{b_n\}\)的前\(n\)项和\(S_n\)。",
            "correct_agents_thinking": ["首先，对\(a_{n + 1}=\frac{a_n}{1 + 3a_n}\)两边取倒数，得到\(\frac{1}{a_{n + 1}}=\frac{1 + 3a_n}{a_n}=\frac{1}{a_n}+3\)，由此可知数列\(\{\frac{1}{a_n}\}\)是首项为\(1\)，公差为\(3\)的等差数列，进而求出\(\frac{1}{a_n}\)的通项公式。然后根据\(b_n=\frac{1}{a_n}+2^n\)求出\(b_n\)的通项公式，\(b_n\)的通项公式是一个等差数列与等比数列相加的形式，利用分组求和法，分别求出等差数列和等比数列的前\(n\)项和，再相加得到\(S_n\)。"]
        },
        {
            "problem": "已知函数\(f(x)=e^x - ax^2 - bx - 1\)，其中\(a\)，\(b\in R\)，\(e = 2.71828\cdots\)为自然对数的底数。若\(f(1)=0\)，且函数\(f(x)\)在区间\((0,1)\)内有零点，求\(a\)的取值范围。",
            "correct_agents_thinking": ["由\(f(1)=0\)可得\(b = e - a - 1\)，将其代入\(f(x)\)。因为函数\(f(x)\)在区间\((0,1)\)内有零点，所以\(f(x)\)在\((0,1)\)内至少存在一个极值点。对\(f(x)\)求导，令\(g(x)=f^\prime(x)\)，则\(g(x)\)在\((0,1)\)内有变号零点。通过对\(g(x)\)求导分析其单调性，结合\(g(0)\)，\(g(1)\)的值以及\(g(x)\)在\((0,1)\)内的单调性，列出关于\(a\)的不等式组，求解不等式组得到\(a\)的取值范围。"]
        }
    ]

    # 调用训练模型函数，这里设置为使用deepseek 1.5b模型进行训练
    start_total_time = time.time()
    trained_model = train_model(test_problem_data_list, model_choice="qwen")
    end_total_time = time.time()
    print(f"整个程序运行耗时: {end_total_time - start_total_time} 秒")


if __name__ == "__main__":
    main()

