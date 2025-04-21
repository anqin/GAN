#!/usr/bin/env python3
# encoding: utf-8
# coding style: pep8
# ====================================================
#   Copyright (C) 2025 ANQIN-X Project. All rights reserved.
#
#   Author        : An Qin
#   Email         : anqin.qin@gmail.com
#   File Name     : sdg_main.py
#   Last Modified : 2025-04-21 16:56
#   Describe      : 
#
# ====================================================


import requests
import json
from agent_manager import agent_list, add_agent
from torch_trainer import train_model
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MATH_PROBLEM = "试求一共有多少个有序正整数对\((a,b)\)使得\(a + b = 1000\)，并且\(a\)和\(b\)的十进制表达式中均不出现数字\(0\)"

def generate_prompt(prompt, api_url, api_key, model):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }
    response = requests.post(api_url, json=data, headers=headers)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise Exception(f"API 请求失败，状态码: {response.status_code}")


# 定义出题的prompt模板
generate_problem_prompt_template = "请生成一道{more_difficult}的数学问题：{last_problem}。题目需涉及代数、几何、概率或微积分中的至少一种知识领域，题目表述清晰，无需给出答案。" if "{last_problem}" in "{last_problem}" else "请生成一道有一定难度的数学问题，涉及代数、几何、概率或微积分中的至少一种知识领域，题目表述清晰，无需给出答案。"
# 定义解题的prompt模板
solve_problem_prompt_template = "请详细解答以下数学问题，给出完整的深度思考路径和答案：{problem}。在思考过程中，请尽可能详细地阐述每一步的推理和依据。"
# 定义判断答案的prompt模板
check_answer_prompt_template = "请判断以下解答是否正确：{answer}，对于该解答，请详细分析是否正确解决了问题：{problem}。如果正确请回复'正确'，如果错误请指出错误之处。"


# 通用出题Agent
def problem_generating_agent(last_problem, api_url, api_key, model, more_difficult="有一定难度"):
    if last_problem:
        prompt = generate_problem_prompt_template.format(more_difficult="比以下题目更难", last_problem=last_problem)
    else:
        prompt = generate_problem_prompt_template.format(more_difficult=more_difficult, last_problem="")
    return generate_prompt(prompt, api_url, api_key, model)


# 通用解题Agent
def problem_solving_agent(problem, api_url, api_key, model):
    prompt = solve_problem_prompt_template.format(problem=problem)
    return generate_prompt(prompt, api_url, api_key, model)


# 通用答案判断Agent
def answer_checking_agent(answer, problem, api_url, api_key, model):
    prompt = check_answer_prompt_template.format(answer=answer, problem=problem)
    return generate_prompt(prompt, api_url, api_key, model)


# 对训练后的custom agent进行评估的函数
def evaluate_trained_custom_agent(custom_agent, agent_list, all_problems_and_answers, num_rounds=5):
    scores = {agent["name"]: 0 for agent in agent_list}
    last_problem = None
    for _ in range(num_rounds):
        for agent in agent_list:
            # 出题
            problem = problem_generating_agent(last_problem, agent["api_url"], agent["api_key"], agent["model"])
            # custom agent作答
            answer = problem_solving_agent(problem, custom_agent["api_url"], custom_agent["api_key"], custom_agent["model"])
            # 其他agent判断答案是否正确
            other_agents = [a for a in agent_list if a["name"] != custom_agent["name"]]
            check_results = [answer_checking_agent(answer, problem, other_agent["api_url"], other_agent["api_key"], other_agent["model"]) for other_agent in other_agents]
            incorrect_count = sum(["错误" in result for result in check_results])
            scores[custom_agent["name"]] += incorrect_count
            last_problem = problem
    return scores[custom_agent["name"]]


# 数学挑战剧本
def math_challenge_script():
    scores = {agent["name"]: 0 for agent in agent_list}
    all_problems_and_answers = []
    last_problem = None #DEFAULT_MATH_PROBLEM
    # 新增开关，用于控制是否开启custom agent的训练和更新后的评估
    enable_custom_agent_training = True
    # 新增开关，用于选择使用deepseek还是qwen进行模型训练
    use_deepseek_for_training = True  

    for _ in range(10):
        for agent in agent_list:
            # 出题
            problem = problem_generating_agent(last_problem, agent["api_url"], agent["api_key"], agent["model"])
            # 其他agent作答
            other_agents = [a for a in agent_list if a["name"] != agent["name"]]
            answers = []
            for other_agent in other_agents:
                answer = problem_solving_agent(problem, other_agent["api_url"], other_agent["api_key"], other_agent["model"])
                answers.append({"agent": other_agent["name"], "answer": answer})
            # 出题者判断答案是否正确
            check_results = [answer_checking_agent(answer["answer"], problem, agent["api_url"], agent["api_key"], agent["model"]) for answer in answers]
            correct_agents = []
            for idx, result in enumerate(check_results):
                if "正确" in result:
                    correct_agents.append(answers[idx]["agent"])
            incorrect_count = sum(["错误" in result for result in check_results])
            scores[agent["name"]] += incorrect_count
            # 保存题目和答案，以及答对的agent的思考过程
            problem_data = {
                "problem": problem,
                "answers": answers,
                "correctness_checks": check_results,
                "correct_agents_thinking": []
            }
            for answer in answers:
                if answer["agent"] in correct_agents:
                    problem_data["correct_agents_thinking"].append(answer["answer"])
            all_problems_and_answers.append(problem_data)
            last_problem = problem

    # 计算custom agent得分
    custom_scores = 0
    custom_agent = None
    for agent in agent_list:
        if agent["name"] == "custom":
            custom_agent = agent
            break
    if custom_agent:
        wrong_problem_data_list = []
        for problem_data in all_problems_and_answers:
            problem = problem_data["problem"]
            answer = problem_solving_agent(problem, custom_agent["api_url"], custom_agent["api_key"], custom_agent["model"])
            check_results = [answer_checking_agent(answer, problem, custom_agent["api_url"], custom_agent["api_key"], custom_agent["model"])]
            if "错误" in check_results[0]:
                wrong_problem_data_list.append(problem_data)
            for check_answer in problem_data["answers"]:
                check_result = answer_checking_agent(answer, problem, custom_agent["api_url"], custom_agent["api_key"], custom_agent["model"])
                if "错误" in check_result:
                    custom_scores += 1
        scores["custom"] = custom_scores

    # 判断custom agent得分是否不是最高且开关开启
    max_score = max(scores.values())
    if custom_scores < max_score and custom_agent and enable_custom_agent_training:
        # 调用torch_trainer.py中的训练函数进行训练，使用答不对的数据，并传递模型选择开关
        trained_model = train_model(wrong_problem_data_list, use_deepseek=use_deepseek_for_training)

        # 使用训练后的模型继续进行评估
        new_custom_score = evaluate_trained_custom_agent(custom_agent, agent_list, all_problems_and_answers)
        scores[custom_agent["name"]] = new_custom_score

    # 保存到json文件
    with open('math_challenge_data.json', 'w', encoding='utf-8') as f:
        json.dump(all_problems_and_answers, f, ensure_ascii=False, indent=4)

    for agent_name, score in scores.items():
        print(f"{agent_name} Agent得分: {score}")


if __name__ == "__main__":
    math_challenge_script()
