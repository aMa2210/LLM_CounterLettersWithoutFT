import csv
import matplotlib.pyplot as plt
from collections import defaultdict

model_names = ['LlaMa3.1-8B_IFCorrect', 'LlaMa3.1-70B_IFCorrect', 'LLama3.2-11B_IFCorrect', 'Mistral-7B_IFCorrect',
               'Mixtral-8x7b_IFCorrect', 'Gemma2-9B_IFCorrect', 'GPT4o-mini_IFCorrect', 'GPT4o_IFCorrect']


def plot_barChart(orderColumn, model_names):
    plt.figure(figsize=(18, 6))
    file_path = 'Results/Random_10000_words.csv'

    # Define the width of the bars
    bar_width = 0.1
    # Set the positions for the bars
    x_positions = range(len(model_names))

    for idx, model_name in enumerate(model_names):
        data = []
        with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data.append([int(row[orderColumn]), int(row[model_name])])

        grouped_data = defaultdict(int)
        grouped_count = defaultdict(int)
        for item in data:
            first_value = item[0]  # Letters_Number
            second_value = 1 - item[1]  # error
            grouped_data[first_value] += second_value
            grouped_count[first_value] += 1

        grouped_data = {key: grouped_data[key] for key in grouped_data if 3 <= key <= 14} # Only retain the data within this range, as the other data samples are too few.
        grouped_count = {key: grouped_count[key] for key in grouped_count if 3 <= key <= 14}

        grouped_average = {key: grouped_data[key] / grouped_count[key] for key in grouped_data}

        keys = list(grouped_average.keys())  # 获取分组的键
        values = list(grouped_average.values())  # 获取每个分组的统计值（即和）

        # Adjust the positions for each model's bars
        offset = bar_width * idx - (len(model_names) - 1) * bar_width / 2  # Calculate the offset for each model
        plt.bar([key + offset for key in keys], values, width=bar_width, label=model_name)
    for key, value in sorted(grouped_count.items()):
        print(f'{key}: {value}')
    fontsize = 16
    plt.xticks(range(3, 15), fontsize=fontsize)
    # plt.xticks(range(1, 33), fontsize=fontsize)
    plt.yticks([i * 0.1 for i in range(11)], fontsize=fontsize)
    plt.xlabel('Number of Letters', fontsize=fontsize)
    plt.ylabel('Percentage of Words with Count Errors', fontsize=fontsize)
    plt.legend(loc='upper left', fontsize=fontsize)
    plt.show()


# plot_barChart('Letters_Number', model_names)
plot_barChart('Letters_Number', model_names)
