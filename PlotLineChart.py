import csv
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
# import numpy as np

model_names = ['LlaMa3.1-8B_IFCorrect', 'LlaMa3.1-70B_IFCorrect', 'LLama3.2-11B_IFCorrect', 'Mistral-7B_IFCorrect',
               'Mixtral-8x7b_IFCorrect', 'Gemma2-9B_IFCorrect', 'GPT4o-mini_IFCorrect', 'GPT4o_IFCorrect']


def plot_lineChart(orderColumn, model_names, x_label='tmp'):
    plt.figure(figsize=(18, 6))
    file_path = 'Results/Random_10000_words.csv'
    # file_path = 'Results/Top_10000_words.csv'

    with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        all_data = []
        for row in reader:
            all_data.append([int(row[orderColumn])] + [int(row[m]) for m in model_names])

    all_data.sort(key=lambda x: x[0], reverse=True)
    order_values = [x[0] for x in all_data]
    change_indices = [0]  # 起始位置
    for i in range(1, len(order_values)):
        if order_values[i] != order_values[i - 1]:
            change_indices.append(i)
    change_indices.append(len(order_values))  # 结束位置



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


    colors = ['#f0f0f0', '#e0e0ff']  # 两种交替颜色
    for i in range(len(change_indices) - 1):
        plt.axvspan(change_indices[i], change_indices[i+1], facecolor=colors[i % len(colors)], alpha=0.6)

    y_top = max(model_values) * 1.02  # 稍微高于最高曲线

    last_plotted_idx=0
    for i in range(len(change_indices) - 1):
        center_idx = (change_indices[i] + change_indices[i + 1]) // 2
        if center_idx < len(data) and (center_idx - last_plotted_idx >= 300):
            plt.text(center_idx, y_top, str(data[center_idx][0]),
                     rotation=0, va='bottom', ha='center', fontsize=16, color='black')
        last_plotted_idx = center_idx

    fontsize = 18
    # plt.xlabel(f'Sorted by {orderColumn}', fontsize=fontsize)
    # plt.xlabel('Words Sorted by Frequency', fontsize=fontsize)
    # plt.xlabel('Words Sorted by Number of Letters', fontsize=fontsize)
    # plt.xlabel('Words Sorted by Difference between Number of Letters and Distinct Letters', fontsize=fontsize)
    plt.xlabel(x_label, fontsize=fontsize)


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


# plot_lineChart('Frequency', model_names, 'Words Sorted by Frequency')
plot_lineChart('Letters_Number',model_names, 'Words Sorted by Number of Letters')
###### plot_lineChart('Unique_Letters_number',model_names)
plot_lineChart('Difference',model_names, 'Words Sorted by Difference between Number of Letters and Distinct Letters')
######## plot_lineChart('Tokens_Number',model_names, )
