import csv
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from collections import defaultdict
from collections import Counter

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


    plot_barChart_letter_several_tokens('Word', model_names, token_file_names)


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


def letters_onlyin_one_token_counts(phrase):
    tokens = phrase.split()
    all_chars = Counter("".join(tokens))
    count = 0
    for token in tokens:

        word_counter = Counter(token)

        if any(word_counter[char] > 1 and all_chars[char] == word_counter[char] for char in word_counter):
            count += 1
    return count

def plot_barChart_letter_several_tokens(orderColumn, model_names, token_file_names):
    plt.figure(figsize=(18, 6))
    file_path = 'Results/Random_10000_words.csv'
    bar_width = 0.1
    model_values = []
    i=0
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
                data_token.append([(row[orderColumn]), letters_onlyin_one_token_counts(row['Subwords'])])


        correct_data = defaultdict(int)
        data_count = defaultdict(int)

        for item_correct in data_correct:
            for item_token in data_token:
                if item_token[0] == item_correct[0]:
                    correct_data[item_token[1]] += (1- item_correct[1])
                    data_count[item_token[1]] += 1
                    continue
        grouped_average = {key: (correct_data[key] / data_count[key]) * 100 for key in correct_data}
        grouped_average = {key: grouped_average[key] for key in grouped_average if 0 <= key <= 2}

        for key in data_count:
            print(f'key:{key}, Value:{data_count[key]}')
        keys = list(grouped_average.keys())
        values = list(grouped_average.values())

        # Adjust the positions for each model's bars
        offset = bar_width * i - (len(model_names) - 1) * bar_width / 2  # Calculate the offset for each model
        plt.bar([key + offset for key in keys], values, width=bar_width, label=model_name)
        i += 1
        print('done**********************************')

    fontsize = 16
    # plt.xticks(range(3, 15), fontsize=fontsize)
    # plt.xticks(range(1, 8), fontsize=fontsize)
    plt.xticks(range(0, 3), fontsize=fontsize)
    plt.yticks([i * 10 for i in range(11)], fontsize=fontsize)
    plt.xlabel('Number of Letters That Appear and Repeat Exclusively Within a Single Token', fontsize=fontsize)
    plt.ylabel('Percentage of Words with Count Errors (%)', fontsize=fontsize)
    plt.legend(loc='upper left', fontsize=fontsize-4)
    plt.show()


if __name__ == "__main__":
    main()
