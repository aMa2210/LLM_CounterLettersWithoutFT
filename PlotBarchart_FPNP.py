import json
import pandas as pd
from collections import Counter
from openpyxl.styles import Font
import matplotlib.pyplot as plt
import numpy as np

def main():
    model_names = ['LlaMa3.1-8B_IFCorrect', 'LlaMa3.1-70B_IFCorrect', 'LLama3.2-11B_IFCorrect', 'Mistral-7B_IFCorrect',
                   'Mixtral-8x7b_IFCorrect', 'Gemma2-9B_IFCorrect', 'GPT4o-mini_IFCorrect', 'GPT4o_IFCorrect']
    model_names_json = ['LlaMa3.1_8B', 'LlaMa3.1_70B', 'LLama3.2_11B', 'mistral_7B',
                        'mixtral-8x7b', 'gemma2-9B', 'gpt4o_mini', 'gpt4o']
    model_names_json = ['Results/' + result + '.json' for result in model_names_json]

    plotResults(model_names, model_names_json)


def letter_count(word, json_word):
    count = {i: {j: 0 for j in range(9)} for i in range(1, 9)}
    count_not_exist = {
        'non_existent': 0,
        'total': 0
    }
    all_chars = Counter(word)
    if isinstance(json_word, dict):
        for char in all_chars:
            if char in json_word:
                count[all_chars[char]][json_word[char]] += 1
                # if all_chars[char] == 8:
                #     print(word)
            else:
                count[all_chars[char]][0] += 1

    # count number of times that the model generate an answer including a letter while it actually doesn't exist in the word
    if isinstance(json_word, dict):
        count_not_exist['total'] +=1
        if any(char not in all_chars for char in json_word):
            count_not_exist['non_existent'] += 1


    return count, count_not_exist


def plotResults(model_names, model_names_json):
    all_model_data = []

    for model_name, model_name_json in zip(model_names, model_names_json):
        with open(model_name_json, "r") as json_file:  # Load JSON format answers
            data_json = json.load(json_file)
        for key, value in data_json.items():  # Parse the JSON-formatted value
            try:
                data_json[key] = json.loads(value)
            except:
                pass

        count = {i: {j: 0 for j in range(9)} for i in range(1, 9)}
        count_not_exist = {
            'non_existent': 0,
            'total': 0
        }

        for word, json_word in data_json.items():
            word_count, word_count_nonExist = letter_count(word, json_word)
            for real_count in word_count:
                for pred_count in word_count[real_count]:
                    count[real_count][pred_count] += word_count[real_count][pred_count]
            for entity in count_not_exist:
                count_not_exist[entity] += word_count_nonExist[entity]

        # Convert the count dictionary to a pandas DataFrame
        df = pd.DataFrame.from_dict(count, orient='index')
        df.index.name = 'Reality\Prediction'
        df.columns = [f'{i}' for i in range(9)]

        # Add a 'Total' column
        df['Total'] = df.sum(axis=1)

        df_filtered = df.loc[1:3]
        correct_1 = df_filtered.loc[1, '1']  # 真实值为 1，预测值为 1 的数量
        less_than_correct_1 = df_filtered.loc[1, '0']  # 预测值小于 1 的情况，只考虑预测值为 0
        greater_than_correct_1 = df_filtered.loc[1, '2'] + df_filtered.loc[1, '3'] + df_filtered.loc[1, '4'] + \
                                 df_filtered.loc[1, '5'] + df_filtered.loc[1, '6'] + df_filtered.loc[1, '7'] + \
                                 df_filtered.loc[1, '8']  # 预测值大于 1 的情况

        # 真实值为 2 时
        correct_2 = df_filtered.loc[2, '2']  # 真实值为 2，预测值为 2 的数量
        less_than_correct_2 = df_filtered.loc[2, '0'] + df_filtered.loc[2, '1']  # 预测值小于 2 的情况，只考虑预测值为 0 和 1
        greater_than_correct_2 = df_filtered.loc[2, '3'] + df_filtered.loc[2, '4'] + df_filtered.loc[2, '5'] + \
                                 df_filtered.loc[2, '6'] + df_filtered.loc[2, '7'] + df_filtered.loc[
                                     2, '8']  # 预测值大于 2 的情况

        # 真实值为 3 时
        correct_3 = df_filtered.loc[3, '3']  # 真实值为 3，预测值为 3 的数量
        less_than_correct_3 = df_filtered.loc[3, '0'] + df_filtered.loc[3, '1'] + df_filtered.loc[3, '2']  # 预测值小于 3 的情况
        greater_than_correct_3 = df_filtered.loc[3, '4'] + df_filtered.loc[3, '5'] + df_filtered.loc[3, '6'] + \
                                 df_filtered.loc[3, '7'] + df_filtered.loc[3, '8']  # 预测值大于 3 的情况
        all_model_data.append({
            'model_name': model_name,
            'correct_1': correct_1,
            'less_than_correct_1': less_than_correct_1,
            'greater_than_correct_1': greater_than_correct_1,
            'correct_2': correct_2,
            'less_than_correct_2': less_than_correct_2,
            'greater_than_correct_2': greater_than_correct_2,
            'correct_3': correct_3,
            'less_than_correct_3': less_than_correct_3,
            'greater_than_correct_3': greater_than_correct_3
        })
    fig, ax = plt.subplots(figsize=(10, 7))

    bar_width = 0.2  # 柱形的宽度

    index = np.arange(len(model_names))  # x轴位置

    labels_set = False

    color1 = (222/255,120/255,51/255)
    color2 = (144/255, 37/255, 37/255)
    color3 = (39/255, 108/255, 158/255)
    # color1 = 'r'
    # color2 = 'g'
    # color3 = 'b'
    # 对每个模型绘制3个柱形
    for idx, model_data in enumerate(all_model_data):

        bottom_1 = 0
        bottom_2 = 0
        bottom_3 = 0

        offset = bar_width * 0.15

        # 为每个模型绘制3个柱形
        ax.bar(index[idx] - bar_width - offset, model_data['correct_1'], bar_width, color=color1,  label='Correct' if not labels_set else "")
        bottom_1 += model_data['correct_1']  # 更新 bottom_1
        ax.bar(index[idx] - bar_width - offset, model_data['less_than_correct_1'], bar_width, bottom=bottom_1, color=color2, label='Lower' if not labels_set else "")
        bottom_1 += model_data['less_than_correct_1']  # 更新 bottom_1
        ax.bar(index[idx] - bar_width - offset, model_data['greater_than_correct_1'], bar_width,bottom=bottom_1, color=color3, label='Higher' if not labels_set else "")
        bottom_1 += model_data['greater_than_correct_1']  # 更新 bottom_1

        ax.bar(index[idx], model_data['correct_2'], bar_width, color=color1)
        bottom_2 += model_data['correct_2']  # 更新 bottom_2
        ax.bar(index[idx], model_data['less_than_correct_2'], bar_width,bottom=bottom_2, color=color2)
        bottom_2 += model_data['less_than_correct_2']  # 更新 bottom_2
        ax.bar(index[idx], model_data['greater_than_correct_2'], bar_width,bottom=bottom_2,  color=color3)
        bottom_2 += model_data['greater_than_correct_2']  # 更新 bottom_2

        ax.bar(index[idx] + bar_width + offset, model_data['correct_3'], bar_width, color=color1)
        bottom_3 += model_data['greater_than_correct_3']  # 更新 bottom_2
        ax.bar(index[idx] + bar_width + offset, model_data['less_than_correct_3'], bar_width, bottom=bottom_3, color=color2)
        bottom_3 += model_data['less_than_correct_3']  # 更新 bottom_2
        ax.bar(index[idx] + bar_width + offset, model_data['greater_than_correct_3'], bar_width,bottom=bottom_3,  color=color3)
        bottom_3 += model_data['greater_than_correct_3']  # 更新 bottom_2


        if not labels_set:
            labels_set = True

    # 添加标签和标题
    fontsize = 14
    ax.set_xlabel('Model', fontsize=fontsize)
    ax.set_ylabel('Letter', fontsize=fontsize)
    # ax.set_title('Error Rates by Model')
    ax.set_xticks(index)
    model_names = [name.replace('_IFCorrect','') for name in model_names]
    plt.yticks(fontsize=fontsize)
    ax.set_xticklabels(model_names, fontsize=fontsize)
    ax.legend(loc='upper right', fontsize=fontsize)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
