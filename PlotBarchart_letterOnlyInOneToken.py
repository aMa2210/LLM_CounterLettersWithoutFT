import csv
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from collections import defaultdict
from collections import Counter
import json
import numpy as np

# import numpy as np   {1: {'wrong': 130, 'total': 50786}, 2: {'wrong': 445, 'total': 2740}, 3: {'wrong': 31, 'total': 121}, 4: {'wrong': 0, 'total': 3}}
def main():
    model_names = ['LlaMa3.1-8B_IFCorrect', 'LlaMa3.1-70B_IFCorrect', 'LLama3.2-11B_IFCorrect', 'Mistral-7B_IFCorrect',
                   'Mixtral-8x7b_IFCorrect', 'Gemma2-9B_IFCorrect', 'GPT4o-mini_IFCorrect', 'GPT4o_IFCorrect']
    model_names_json = ['LlaMa3.1_8B', 'LlaMa3.1_70B', 'LLama3.2_11B', 'mistral_7B',
                   'mixtral-8x7b', 'gemma2-9B', 'gpt4o_mini', 'gpt4o']
    model_names_json = ['Results/' + result + '.json' for result in model_names_json]
    token_file_names = ['test_words_tokenized_by_Llama-3.1-8B.csv', 'test_words_tokenized_by_Llama-3.1-8B.csv',
                        'test_words_tokenized_by_Llama-3.1-8B.csv', 'test_words_tokenized_by_Mistral-7B-v0.1.csv',
                        'test_words_tokenized_by_Mistral-7B-v0.1.csv', 'test_words_tokenized_by_gemma-2-9B.csv',
                        'test_words_tokenized_by_gpt-4o-mini.csv', 'test_words_tokenized_by_gpt-4-mini.csv']
    token_file_names = ['Results/token_results/' + file_name for file_name in token_file_names]

    plot_barChart_letter_several_tokens('Word', model_names, token_file_names, model_names_json)

    # phrase = "hello word"
    # json_word = {'h': 1, 'e': 1, 'l': 3, 'o': 1, 'w': 1, 'r': 1, 'd': 1}
    #
    # result = letters_onlyin_one_token_counts(phrase, json_word)
    # print(result)


def letters_onlyin_one_token_counts(phrase, json_word, ifprint):
    # print(f"At the start of the function, json_word type: {type(json_word)}")
    # print(json_word)
    tokens = phrase.split()
    all_chars = Counter("".join(tokens))
    count = {
        1: {'wrong': 0, 'total': 0},
        '2_single': {'wrong': 0, 'total': 0},
        '2_multi': {'wrong': 0, 'total': 0},
        # 2: {'wrong': 0, 'total': 0},
        3: {'wrong': 0, 'total': 0},
        4: {'wrong': 0, 'total': 0},
    }
    if isinstance(json_word, dict):
        processed_chars = set()
        for token in tokens:
            token_counter = Counter(token)
            for char in token_counter:
                if char in processed_chars:
                    continue
                else:
                    if all_chars[char] == 2:
                        if all_chars[char] == token_counter[char]:
                            count['2_single']['total'] += 1
                            if char in json_word:
                                if token_counter[char] != json_word.get(char, 0):
                                    count['2_single']['wrong'] += 1

                        else:
                            count['2_multi']['total'] += 1
                            if char in json_word:
                                if all_chars[char] != json_word.get(char, 0):
                                    count['2_multi']['wrong'] += 1
                                    # if ifprint:
                                        # if phrase == 'pan ne ko ek':
                                        #     print(char)
                                        #     print(json_word)
                                        #     print('phrase: ' + phrase)


                    elif all_chars[char] == token_counter[char]:
                        count[token_counter[char]]['total'] += 1
                        # try:
                        if char in json_word:
                            if token_counter[char] != json_word.get(char, 0):
                                count[token_counter[char]]['wrong'] += 1

                    processed_chars.add(char)
                                # if(token_counter[char] == 4):
                                #     print(phrase)
                        # except:
                        #     print(f"json_word keys: {json_word}")
                        #     print(f"char: '{char}'")
                        #     print(f"char in json_word: {char in json_word}")
                        #     print(11)

    return count        #return a dict {1:{wrong:a, total:b}, 2:{wrong:c, total:d} ....}

def plot_barChart_letter_several_tokens(orderColumn, model_names, token_file_names, model_names_json):
    plt.figure(figsize=(18, 6))
    file_path = 'Results/Random_10000_words.csv'
    bar_width = 0.1
    i=0
    values_history = []
    for model_name, token_file_name, model_name_json in zip(model_names, token_file_names, model_names_json):

        with open(model_name_json, "r") as json_file:  #load json format answer i.e letter-level
            data_json = json.load(json_file)
        for key, value in data_json.items():  # Parse the JSON-formatted value in file a from a string into a dictionary to compare
            try:
                data_json[key] = json.loads(value)
            except:
                # print('errorKey: ' + key)
                pass
        data_token = []
        count = {
            1: {'wrong': 0, 'total': 0},
            '2_single': {'wrong': 0, 'total': 0},
            '2_multi': {'wrong': 0, 'total': 0},
            # 2: {'wrong': 0, 'total': 0},
            3: {'wrong': 0, 'total': 0},
            4: {'wrong': 0, 'total': 0},
        }
        with open(token_file_name, 'r', newline='', encoding='utf-8') as csvfile1:
            reader = csv.DictReader(csvfile1)
            for row in reader:
                json_word = data_json.get(row[orderColumn])
                if model_name == 'GPT4o_IFCorrect':
                    ifplot = True
                else:
                    ifplot = False
                dict1 = letters_onlyin_one_token_counts(row['Subwords'], json_word, ifplot)
                for key, sub_dict in dict1.items():
                    for sub_key, value in sub_dict.items():
                        count[key][sub_key] += value
                data_token.append([(row[orderColumn]), ])
        print(count)
        result = {key: (count[key]['wrong'] / count[key]['total']) * 100 for key in count}
        result = {
            key: result[key]
            for key in result
            if key in ['2_single', '2_multi']
        }

        values = list(result.values())
        values_history.append(values)

    values_history = np.array(values_history)
    print(values_history)
    label_name = ['LlaMa3.1-8B_IFCorrect', 'LlaMa3.1-70B_IFCorrect', 'LLama3.2-11B_IFCorrect', 'Mistral-7B_IFCorrect',
                'Mixtral-8x7b_IFCorrect', 'Gemma2-9B_IFCorrect', 'GPT4o-mini_IFCorrect', 'GPT4o_IFCorrect']
    x_labels = [x.replace('_IFCorrect','') for x in label_name]
    x = np.arange(len(label_name))  # 每组柱状图的X轴位置
    bar_width = 0.35  # 每个柱子的宽度

    color_A = 'royalblue'  # 为 Single token 设置颜色
    color_B = 'darkorange'  # 为 Multi token 设置颜色

    # 绘制 Single token 的所有柱子
    plt.bar(x - bar_width / 2, values_history[:, 0], width=bar_width, label='Same token', color=color_A)

    # 绘制 Multi token 的所有柱子
    plt.bar(x + bar_width / 2, values_history[:, 1], width=bar_width, label='Different tokens', color=color_B)

    fontsize = 16
    # plt.xticks(range(3, 15), fontsize=fontsize)
    # plt.xticks(range(1, 8), fontsize=fontsize)
    plt.xticks(range(0, 8),x_labels, fontsize=fontsize)
    plt.yticks([i * 10 for i in range(11)], fontsize=fontsize)
    plt.xlabel('Model Name', fontsize=fontsize)
    plt.ylabel('Percentage of Letters with Count Errors (%)', fontsize=fontsize)
    plt.legend(loc='upper left', fontsize=fontsize)
    plt.show()


if __name__ == "__main__":
    main()
