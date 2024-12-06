import csv
import matplotlib.pyplot as plt
from collections import defaultdict
from collections import Counter


def main():
    model_names = ['LlaMa3.1_8B', 'LlaMa3.1_70B', 'LLama3.2_11B', 'mistral_7B', 'mixtral-8x7b', 'gemma2-9B',
                   'gpt4o_mini', 'gpt4o']

    letter_frequencies = {
        "E": 12.49, "T": 9.28, "A": 8.04, "O": 7.64, "I": 7.57, "N": 7.23, "S": 6.51, "R": 6.28, "H": 5.05, "L": 4.07,
        "D": 3.82, "C": 3.34,
        "U": 2.73, "M": 2.51, "F": 2.40, "P": 2.14, "G": 1.87, "W": 1.68, "Y": 1.66, "B": 1.48, "V": 1.05, "K": 0.54,
        "X": 0.23, "J": 0.16,
        "Q": 0.12, "Z": 0.09
    }
    plot_barChart('Letter', model_names, letter_frequencies)


def plot_barChart(orderColumn, model_names, letter_frequencies):
    plt.figure(figsize=(18, 6))
    file_path = 'Results/Letter_Accuracy.csv'

    bar_width = 0.1
    # Set the positions for the bars
    x_positions = range(len(model_names))
    sorted_letters = sorted(letter_frequencies.keys(), key=lambda x: letter_frequencies[x], reverse=True)
    data = []
    with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row)


    for idx, model_name in enumerate(model_names):

        letter_accuracies = {}
        for item in data:
            letter_accuracies[item['Letter']] = 100-float(item[model_name])

        # keys = list(letter_accuracies.keys())

        sorted_letter_accuracies = {letter.lower(): letter_accuracies.get(letter.lower(), 0) for letter in sorted_letters}
        values = [accuracy for accuracy in sorted_letter_accuracies.values()]

        offset = bar_width * idx - (len(model_names) - 1) * bar_width / 2
        x_positions_for_bars = [i + offset for i in range(len(sorted_letters))]

        plt.bar(x_positions_for_bars, values, width=bar_width, label=model_name)

    for key, value in sorted(letter_accuracies.items()):
        print(f'{key}: {value}')
    fontsize = 16

    plt.xticks(fontsize=fontsize)
    plt.xlim(-0.5, len(sorted_letters) - 0.5)
    plt.xticks([x - 0.37 for x in x_positions_for_bars], sorted_letter_accuracies.keys(), fontsize=fontsize)
    plt.yticks([i * 10 for i in range(5)], fontsize=fontsize)
    plt.xlabel('Letters Sorted by Frequency', fontsize=fontsize)
    plt.ylabel('Percentage of Letters with Count Errors (%)', fontsize=fontsize)
    plt.legend(loc='best', fontsize=fontsize - 6)

    plt.show()


if __name__ == "__main__":
    main()
