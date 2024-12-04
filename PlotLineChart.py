import csv
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
# import numpy as np

model_names = ['LlaMa3.1-8B_IFCorrect', 'LlaMa3.1-70B_IFCorrect', 'LLama3.2-11B_IFCorrect', 'Mistral-7B_IFCorrect',
               'Mixtral-8x7b_IFCorrect', 'Gemma2-9B_IFCorrect', 'GPT4o-mini_IFCorrect', 'GPT4o_IFCorrect']


def plot_lineChart(orderColumn, model_names):
    plt.figure(figsize=(18, 6))
    file_path = 'Results/Random_10000_words.csv'
    # file_path = 'Results/Top_10000_words.csv'
    model_values = []
    for model_name in model_names:
        data = []
        with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data.append([int(row[orderColumn]), int(row[model_name])])

        data.sort(key=lambda x: x[0], reverse=True)
        second_column = [1 - x[1] for x in data]  # calculate error counts

        spearman_corr, _ = spearmanr([x[0] for x in data], second_column)
        print(f'Spearman correlation for {model_name}: {spearman_corr}')

        cumulative_values = [sum(second_column[:i + 1]) for i in range(len(second_column))]
        plt.plot(cumulative_values, label=model_name.replace('_IFCorrect', ''), linewidth=6)
        model_values.append(max(cumulative_values))
    # print(len(cumulative_values))
    fontsize = 12
    # plt.xlabel(f'Sorted by {orderColumn}', fontsize=fontsize)
    # plt.xlabel('Words Sorted by Frequency', fontsize=fontsize)
    # plt.xlabel('Words Sorted by Number of Letters', fontsize=fontsize)
    plt.xlabel('Words Sorted by Difference between Number of Letters and Distinct Letters', fontsize=fontsize)


    plt.ylabel('Cumulative Errors', fontsize=fontsize)
    # plt.title(f'Random words_{model_name.strip("_IFCorrect")}_{orderColumn}', fontsize=fontsize)
    plt.xticks([i * 1000 for i in range(11)], fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlim(0, len(cumulative_values))
    plt.ylim(0, max(model_values))
    plt.grid(False)
    plt.tick_params(axis='both', direction='in', length=6, width=1)
    plt.legend(loc='upper left')
    plt.show()


# plot_lineChart('Frequency', model_names)
# plot_lineChart('Letters_Number',model_names)
# plot_lineChart('Unique_Letters_number',model_names)
plot_lineChart('Difference',model_names)
# plot_lineChart('Tokens_Number',model_names)
