#!/usr/bin/env python3
# encoding: utf-8
# coding style: pep8
# ====================================================
#   Copyright (C) 2025 ANQIN-X Project. All rights reserved.
#
#   Author        : An Qin
#   Email         : anqin.qin@gmail.com
#   File Name     : agent_manager.py
#   Last Modified : 2025-04-21 16:53
#   Describe      : 
#
# ====================================================

import sys
# import os


# 定义一个列表，用于存储agent的相关信息，每个元素是一个字典，包含API地址、API密钥、模型名称和agent名称
agent_list = [
    {
        "name": "deepseek",
        "api_url": "https://api.deepseek.com/v1/chat/completions",
        "api_key": "your_deepseek_api_key",
        "model": "deepseek/v1.0"
    },
    {
        "name": "qwen",
        "api_url": "https://your_qwen_api_url/v1/chat/completions",
        "api_key": "your_qwen_api_key",
        "model": "qwen_model_name"
    },
    {
        "name": "gpt4",
        "api_url": "https://your_gpt4_api_url/v1/chat/completions",
        "api_key": "your_gpt4_api_key",
        "model": "gpt-4"
    },
    {
        "name": "gemini",
        "api_url": "https://your_gemini_api_url/v1/chat/completions",
        "api_key": "your_gemini_api_key",
        "model": "gemini_model_name"
    },
    {
        "name": "doubao",
        "api_url": "https://your_doubao_api_url/v1/chat/completions",
        "api_key": "your_doubao_api_key",
        "model": "doubao-1.5"
    }
]

# 可扩展添加新agent的函数
def add_agent(name, api_url, api_key, model):
    new_agent = {
        "name": name,
        "api_url": api_url,
        "api_key": api_key,
        "model": model
    }
    agent_list.append(new_agent)
    print(f"已成功添加agent: {name}")
