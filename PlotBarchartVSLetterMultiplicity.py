import csv
import matplotlib.pyplot as plt
from collections import defaultdict
from collections import Counter


def main():
    model_names = ['LlaMa3.1-8B_IFCorrect', 'LlaMa3.1-70B_IFCorrect', 'LLama3.2-11B_IFCorrect', 'Mistral-7B_IFCorrect',
                   'Mixtral-8x7b_IFCorrect', 'Gemma2-9B_IFCorrect', 'GPT4o-mini_IFCorrect', 'GPT4o_IFCorrect']
    plot_barChart('Word', model_names)


def max_letter_count(word):
    letter_counts = Counter(word)
    # return frequency of the most common letter
    return max(letter_counts.values())


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
                data.append([int(max_letter_count(row[orderColumn])), int(row[model_name])])

        grouped_data = defaultdict(int)
        grouped_count = defaultdict(int)
        for item in data:
            first_value = item[0]  # Letters_Number
            second_value = 1 - item[1]  # error
            grouped_data[first_value] += second_value
            grouped_count[first_value] += 1

        grouped_data = {key: grouped_data[key] for key in grouped_data if
                        1 <= key <= 4}  # Only retain the data within this range, as the other data samples are too few.
        grouped_count = {key: grouped_count[key] for key in grouped_count if 1 <= key <= 4}
        print(grouped_data)
        print(grouped_count)
        print('********************')
        grouped_average = {key: (grouped_data[key] / grouped_count[key]) * 100 for key in grouped_data}

        keys = list(grouped_average.keys())
        values = list(grouped_average.values())

        # Adjust the positions for each model's bars
        offset = bar_width * idx - (len(model_names) - 1) * bar_width / 2  # Calculate the offset for each model
        plt.bar([key + offset for key in keys], values, width=bar_width, label=model_name)
    for key, value in sorted(grouped_count.items()):
        print(f'{key}: {value}')
    fontsize = 16
    # plt.xticks(range(3, 15), fontsize=fontsize)
    # plt.xticks(range(1, 8), fontsize=fontsize)
    plt.xticks(range(1, 5), fontsize=fontsize)
    plt.yticks([i * 10 for i in range(11)], fontsize=fontsize)
    plt.xlabel('Letter Multiplicity', fontsize=fontsize)
    plt.ylabel('Percentage of Words with Count Errors (%)', fontsize=fontsize)
    plt.legend(loc='upper left', fontsize=fontsize)
    plt.show()


if __name__ == '__main__':
    main()
