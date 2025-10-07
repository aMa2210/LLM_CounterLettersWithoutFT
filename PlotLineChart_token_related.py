import csv
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from collections import Counter
from tqdm import tqdm

# import numpy as np
def main():
    model_names = ['LlaMa3.1-8B_IFCorrect', 'LlaMa3.1-70B_IFCorrect', 'LLama3.2-11B_IFCorrect', 'Mistral-7B_IFCorrect',
                   'Mixtral-8x7b_IFCorrect', 'Gemma2-9B_IFCorrect', 'GPT4o-mini_IFCorrect', 'GPT4o_IFCorrect']
    token_file_names = ['test_words_tokenized_by_Llama-3.1-8B.csv', 'test_words_tokenized_by_Llama-3.1-8B.csv',
                        'test_words_tokenized_by_Llama-3.1-8B.csv', 'test_words_tokenized_by_Mistral-7B-v0.1.csv',
                        'test_words_tokenized_by_Mistral-7B-v0.1.csv', 'test_words_tokenized_by_gemma-2-9B.csv',
                        'test_words_tokenized_by_gpt-4o-mini.csv', 'test_words_tokenized_by_gpt-4-mini.csv']
    token_file_names = ['Results/token_results/' + file_name for file_name in token_file_names]
    token_frequency_files = ['count_1w_frequency_tokens_tokenized_by_Llama-3.1-8B.csv',
                             'count_1w_frequency_tokens_tokenized_by_Llama-3.1-8B.csv',
                             'count_1w_frequency_tokens_tokenized_by_Llama-3.1-8B.csv',
                             'count_1w_frequency_tokens_tokenized_by_Mistral-7B-v0.1.csv',
                             'count_1w_frequency_tokens_tokenized_by_Mistral-7B-v0.1.csv',
                             'count_1w_frequency_tokens_tokenized_by_gemma-2-9b.csv',
                             'count_1w_frequency_tokens_tokenized_by_gpt-4o-mini.csv',
                             'count_1w_frequency_tokens_tokenized_by_gpt-4-mini.csv']
    token_frequency_files = ['Results/token_results/' + file_name for file_name in token_frequency_files]
    # plot_lineChart_token_num('Word', model_names, token_file_names)
    plot_lineChart_letter_several_tokens('Word', model_names, token_file_names)
    # plot_lineChart_token_frequency('Word', model_names, token_file_names, token_frequency_files)
    # print(inter_token_counts('not und el end'))


def count_Token_Average_Frequency(tokens, frequency_file):
    tokens = tokens.split()
    frequency_dict = {}
    with open(frequency_file, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            frequency_dict[row['Token']] = int(row['Frequency'])

    total_frequency = 0
    tokens_count = 0
    for token in tokens:
        tokens_count += 1
        if token in frequency_dict:
            total_frequency += frequency_dict[token]

        else:
            raise ValueError(f'token:{token} is not in the dict')
    return total_frequency/tokens_count


def save_to_csv(data, output_file):

    # 将数据写入 CSV 文件
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        # 获取字段名，假设 row 是字典，取其中的字段名作为 CSV 文件的标题
        fieldnames = list(data[0][0].keys()) + ['Average_Frequency']  # 在原有字段名后追加 'Frequency' 字段

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        # 写入每一行数据
        for row, a in data:
            # 在原字典中添加 'Frequency' 键值对
            row['Average_Frequency'] = a
            writer.writerow(row)


def plot_lineChart_token_frequency(orderColumn, model_names, token_file_names, token_frequency_files):
    plt.figure(figsize=(18, 6))
    file_path = 'Results/Random_10000_words.csv'
    model_values = []
    for model_name, token_file_name, token_frequency_file in zip(model_names, token_file_names, token_frequency_files):
        data_correct = []
        with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data_correct.append([(row[orderColumn]), int(row[model_name])])
        data_token = []
        data_toSave = []
        with open(token_file_name, 'r', newline='', encoding='utf-8') as csvfile1:
            reader = csv.DictReader(csvfile1)
            for row in tqdm(reader):
                a = count_Token_Average_Frequency(row['Subwords'], token_frequency_file)
                data_token.append([(row[orderColumn]), a])
                data_toSave.append([row, a])
        save_to_csv(data_toSave, token_frequency_file.replace('.csv', 'with_average_frequency.csv'))
        data = []
        for item_correct in data_correct:
            for item_token in data_token:
                if item_token[0] == item_correct[0]:
                    data.append([item_token[1], item_correct[1]])
                    continue

        data.sort(key=lambda x: x[0], reverse=True)
        second_column = [1 - x[1] for x in data]  # calculate error counts

        spearman_corr, _ = spearmanr([x[0] for x in data], second_column)
        print(f'Spearman correlation for {model_name}: {spearman_corr}')

        cumulative_values = [sum(second_column[:i + 1]) for i in range(len(second_column))]
        plt.plot(cumulative_values, label=model_name.replace('_IFCorrect', ''), linewidth=6)
        model_values.append(max(cumulative_values))
    # print(len(cumulative_values))
    fontsize = 18
    # plt.xlabel(f'Sorted by {orderColumn}', fontsize=fontsize)
    # plt.xlabel('Words Sorted by Frequency', fontsize=fontsize)
    # plt.xlabel('Words Sorted by Number of Letters', fontsize=fontsize)
    plt.xlabel('Words Sorted by Average Frequency of Tokens', fontsize=fontsize)

    plt.ylabel('Cumulative Errors', fontsize=fontsize)
    # plt.title(f'Random words_{model_name.strip("_IFCorrect")}_{orderColumn}', fontsize=fontsize)
    plt.xticks([i * 1000 for i in range(11)], fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlim(0, len(cumulative_values))
    plt.ylim(0, max(model_values))
    plt.grid(False)
    plt.tick_params(axis='both', direction='in', length=6, width=1)
    plt.legend(loc='upper left', fontsize=fontsize)
    plt.show()


def inter_token_counts(phrase):
    tokens = phrase.split()
    char_token_map = {}
    for token in tokens:
        for char in set(token):  # 使用set保证每个字符只算一次
            if char not in char_token_map:
                char_token_map[char] = set()
            char_token_map[char].add(token)
    repeated_count = 0
    unique_repeated_count = 0
    for char, token_set in char_token_map.items():
        if len(token_set) > 1:  # 字符在多个token中出现
            for token_1 in token_set:
                repeated_count += token_1.count(char)
            unique_repeated_count += 1
    return repeated_count-unique_repeated_count
def plot_lineChart_letter_several_tokens(orderColumn, model_names, token_file_names):
    plt.figure(figsize=(18, 6))
    file_path = 'Results/Random_10000_words.csv'


    model_values = []
    for model_name, token_file_name in zip(model_names, token_file_names):
        data_correct = []
        with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data_correct.append([(row[orderColumn]), int(row[model_name])])
        data_token = []
        with open(token_file_name, 'r', newline='', encoding='utf-8') as csvfile1:
            reader = csv.DictReader(csvfile1)
            for row in reader:
                data_token.append([(row[orderColumn]), inter_token_counts(row['Subwords'])])
        data = []
        for item_correct in data_correct:
            for item_token in data_token:
                if item_token[0] == item_correct[0]:
                    data.append([item_token[1], item_correct[1]])
                    continue

        data.sort(key=lambda x: x[0], reverse=True)
        second_column = [1 - x[1] for x in data]  # calculate error counts

        spearman_corr, _ = spearmanr([x[0] for x in data], second_column)
        print(f'Spearman correlation for {model_name}: {spearman_corr}')
        cumulative_values = [sum(second_column[:i + 1]) for i in range(len(second_column))]
        if 'change_indices' not in locals() or not change_indices:
            change_indices = [0]  # 起始位置
            for i in range(1, len(data)):
                # 如果 data 中的关键值（比如 token 数量或其他排序依据）发生变化，就记录索引
                if data[i][0] != data[i - 1][0]:
                    change_indices.append(i)
            change_indices.append(len(data))  # 结束位置

        plt.plot(cumulative_values, label=model_name.replace('_IFCorrect', ''), linewidth=6)
        model_values.append(max(cumulative_values))
    # print(len(cumulative_values))
    #########################
    colors = ['#f0f0f0', '#e0e0ff']  # 两种交替颜色
    for i in range(len(change_indices) - 1):
        plt.axvspan(change_indices[i], change_indices[i + 1],
                    facecolor=colors[i % len(colors)], alpha=0.6)

    y_top = max(model_values) * 1.02  # 稍微高于最高曲线
    last_plotted_idx=0
    for i in range(len(change_indices) - 1):
        center_idx = (change_indices[i] + change_indices[i + 1]) // 2
        if center_idx < len(data) and (center_idx - last_plotted_idx >= 300):
            plt.text(center_idx, y_top, str(data[center_idx][0]),
                     rotation=0, va='bottom', ha='center', fontsize=16, color='black')
        last_plotted_idx = center_idx

    ###########################


    fontsize = 18
    # plt.xlabel(f'Sorted by {orderColumn}', fontsize=fontsize)
    # plt.xlabel('Words Sorted by Frequency', fontsize=fontsize)
    # plt.xlabel('Words Sorted by Number of Letters', fontsize=fontsize)
    plt.xlabel('Words Sorted by Difference between Letters and Distinct Letters Appearing Across tokens', fontsize=fontsize)

    plt.ylabel('Cumulative Errors', fontsize=fontsize)
    # plt.title(f'Random words_{model_name.strip("_IFCorrect")}_{orderColumn}', fontsize=fontsize)
    plt.xticks([i * 1000 for i in range(11)], fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlim(0, len(cumulative_values))
    plt.ylim(0, max(model_values))
    plt.grid(False)
    plt.tick_params(axis='both', direction='in', length=6, width=1)
    plt.legend(loc='upper left', fontsize=fontsize)
    plt.show()

def plot_lineChart_token_num(orderColumn, model_names, token_file_names):
    plt.figure(figsize=(18, 6))
    file_path = 'Results/Random_10000_words.csv'

    ##################
    token_transitions = []
    with open(token_file_names[0], 'r', newline='', encoding='utf-8') as csvfile1:
        reader = csv.DictReader(csvfile1)
        for row in reader:
            token_transitions.append(int(row['Number of Tokens']))

    token_transitions.sort(reverse=True)
    # 找出 token 数变化点
    change_indices = [0]
    for i in range(1, len(token_transitions)):
        if token_transitions[i] != token_transitions[i - 1]:
            change_indices.append(i)
    change_indices.append(len(token_transitions))
    ######################


    model_values = []
    for model_name, token_file_name in zip(model_names, token_file_names):
        data_correct = []
        with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data_correct.append([(row[orderColumn]), int(row[model_name])])
        data_token = []
        with open(token_file_name, 'r', newline='', encoding='utf-8') as csvfile1:
            reader = csv.DictReader(csvfile1)
            for row in reader:
                data_token.append([(row[orderColumn]), int(row['Number of Tokens'])])
        data = []
        for item_correct in data_correct:
            for item_token in data_token:
                if item_token[0] == item_correct[0]:
                    data.append([item_token[1], item_correct[1]])
                    continue

        data.sort(key=lambda x: x[0], reverse=True)
        second_column = [1 - x[1] for x in data]  # calculate error counts

        spearman_corr, _ = spearmanr([x[0] for x in data], second_column)
        print(f'Spearman correlation for {model_name}: {spearman_corr}')

        cumulative_values = [sum(second_column[:i + 1]) for i in range(len(second_column))]
        plt.plot(cumulative_values, label=model_name.replace('_IFCorrect', ''), linewidth=6)
        model_values.append(max(cumulative_values))
    # print(len(cumulative_values))
    #########################
    colors = ['#f0f0f0', '#e0e0ff']  # 两种交替颜色
    for i in range(len(change_indices) - 1):
        plt.axvspan(change_indices[i], change_indices[i + 1],
                    facecolor=colors[i % len(colors)], alpha=0.6)
    y_top = max(model_values) * 1.02  # 稍微高于最高曲线
    last_plotted_idx=0
    for i in range(len(change_indices) - 1):
        center_idx = (change_indices[i] + change_indices[i + 1]) // 2
        if center_idx < len(data) and (center_idx - last_plotted_idx >= 300):
            plt.text(center_idx, y_top, str(data[center_idx][0]),
                     rotation=0, va='bottom', ha='center', fontsize=16, color='black')
            last_plotted_idx = center_idx
    ###########################
    fontsize = 18
    # plt.xlabel(f'Sorted by {orderColumn}', fontsize=fontsize)
    # plt.xlabel('Words Sorted by Frequency', fontsize=fontsize)
    # plt.xlabel('Words Sorted by Number of Letters', fontsize=fontsize)
    plt.xlabel('Words Sorted by Number of Tokens', fontsize=fontsize)

    plt.ylabel('Cumulative Errors', fontsize=fontsize)
    # plt.title(f'Random words_{model_name.strip("_IFCorrect")}_{orderColumn}', fontsize=fontsize)
    plt.xticks([i * 1000 for i in range(11)], fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlim(0, len(cumulative_values))
    plt.ylim(0, max(model_values))
    plt.grid(False)
    plt.tick_params(axis='both', direction='in', length=6, width=1)
    plt.legend(loc='upper left', fontsize=fontsize)
    plt.show()


if __name__ == "__main__":
    main()
