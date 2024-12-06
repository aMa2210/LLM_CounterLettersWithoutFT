import json
import string
import csv
import os
import re
import matplotlib.pyplot as plt


def main():
    model_names = ['LlaMa3.1_8B', 'LlaMa3.1_70B', 'LLama3.2_11B', 'mistral_7B',
                   'mixtral-8x7b', 'gemma2-9B', 'gpt4o_mini', 'gpt4o']
    result_files = ['Results/' + result + '.json' for result in model_names]
    validation_file = 'Words/Random_10000_words_letterCount.json'

    By_mul_occurance(result_files, model_names, validation_file)
    # Byaccuracy(result_files, model_names, validation_file)


def By_mul_occurance(result_files, model_names, validation_file):
    i = 0
    num_correct = 0
    letter_stats = {letter: {"wrong": 0, "total": 0, "multiple": 0} for letter in string.ascii_lowercase}
    for result_file, model_name in zip(result_files, model_names):
        # match = re.search(r'(_[^.]+)\.json', result_file)
        #
        # model_name = text+match.group(1)

        with open(result_file, "r") as file_a:
            data_a = json.load(file_a)

        with open(validation_file, "r") as file_b:
            data_b = json.load(file_b)

        for key, value in data_a.items():  # Parse the JSON-formatted value in file a from a string into a dictionary to compare
            try:
                data_a[key] = json.loads(value)
            except:
                # print('errorKey: ' + key)
                pass

        for key in data_a:
            if key in data_b:
                value_a = data_a[key]
                value_b = data_b[key]
                if value_a == value_b:
                    num_correct += 1

                if isinstance(value_a, dict):
                    for letter in string.ascii_lowercase:
                        if letter in value_b:
                            letter_stats[letter]["total"] += 1
                            if letter not in value_a or value_a[letter] != value_b[letter]:
                                letter_stats[letter]["wrong"] += 1
            else:
                print(f"Key '{key}' exists only in file a")
        for key in data_a:
            if key in data_b:
                value_countMultiple = data_b[key]
                for letter in string.ascii_lowercase:
                    if letter in value_countMultiple:
                        if value_countMultiple[letter] > 1:
                            letter_stats[letter]["multiple"] += 1
        # print(result_file + ' accuracy: ' + str(num_correct / 10000))

        accuracy_data = {}

        multiple_frequency = {}
        for letter, stats in letter_stats.items():
            multiple_frequency[letter] = stats['multiple'] / stats['total']

        sorted_multiple_frequency = dict(sorted(multiple_frequency.items(), key=lambda item: item[1], reverse=True))

        for letter, stats in letter_stats.items():
            if stats["total"] > 0:
                accuracy = stats["wrong"] / stats["total"]
                accuracy_data[letter] = round(accuracy * 100, 2)
                # print(f"'{letter}' Occurrence: {stats['multiple']/stats['total']}")
            else:
                accuracy_data[letter] = None
                # print(f"'{letter}' accuracy: No data available")

        sorted_accuracy_data = {letter: accuracy_data[letter] for letter in sorted_multiple_frequency.keys()}

        bar_width = 0.1

        keys = list(sorted_accuracy_data.keys())
        values = list(sorted_accuracy_data.values())

        offset = bar_width * i - (len(model_names) - 1) * bar_width / 2  # Calculate the offset for each model
        x_positions_for_bars = [i + offset for i in range(len(sorted_accuracy_data))]

        plt.bar(x_positions_for_bars, values, width=bar_width, label=model_name)
        i += 1
        num_correct = 0
        letter_stats = {letter: {"wrong": 0, "total": 0, "multiple": 0} for letter in string.ascii_lowercase}

    fontsize = 16
    plt.xlim(-0.5, len(sorted_accuracy_data) - 0.5)
    plt.xticks([x - 0.37 for x in x_positions_for_bars], sorted_accuracy_data.keys(), fontsize=fontsize)
    plt.yticks([i * 10 for i in range(5)], fontsize=fontsize)
    plt.xlabel('Letters Sorted by Multiple Occurrence Frequency', fontsize=fontsize)
    plt.ylabel('Percentage of Letters with Count Errors (%)', fontsize=fontsize)
    plt.legend(loc='best', fontsize=fontsize - 6)
    plt.show()


def Byaccuracy(result_files, model_names, validation_file):
    letter_frequencies = {
        "E": 12.49, "T": 9.28, "A": 8.04, "O": 7.64, "I": 7.57, "N": 7.23, "S": 6.51, "R": 6.28, "H": 5.05, "L": 4.07,
        "D": 3.82, "C": 3.34,
        "U": 2.73, "M": 2.51, "F": 2.40, "P": 2.14, "G": 1.87, "W": 1.68, "Y": 1.66, "B": 1.48, "V": 1.05, "K": 0.54,
        "X": 0.23, "J": 0.16,
        "Q": 0.12, "Z": 0.09
    }
    i = 0
    num_correct = 0
    letter_stats = {letter: {"wrong": 0, "total": 0} for letter in string.ascii_lowercase}
    for result_file, model_name in zip(result_files, model_names):
        # match = re.search(r'(_[^.]+)\.json', result_file)
        #
        # model_name = text+match.group(1)

        with open(result_file, "r") as file_a:
            data_a = json.load(file_a)

        with open(validation_file, "r") as file_b:
            data_b = json.load(file_b)

        for key, value in data_a.items():  # Parse the JSON-formatted value in file a from a string into a dictionary to compare
            try:
                data_a[key] = json.loads(value)
            except:
                # print('errorKey: ' + key)
                pass

        for key in data_a:
            if key in data_b:
                value_a = data_a[key]
                value_b = data_b[key]
                if value_a == value_b:
                    num_correct += 1

                if isinstance(value_a, dict):
                    for letter in string.ascii_lowercase:
                        if letter in value_b:
                            letter_stats[letter]["total"] += 1
                            if letter not in value_a or value_a[letter] != value_b[letter]:
                                letter_stats[letter]["wrong"] += 1
            else:
                print(f"Key '{key}' exists only in file a")
        # print(result_file + ' accuracy: ' + str(num_correct / 10000))

        accuracy_data = {}

        for letter, stats in letter_stats.items():
            if stats["total"] > 0:
                accuracy = stats["wrong"] / stats["total"]
                accuracy_data[letter] = round(accuracy * 100, 2)
            else:
                accuracy_data[letter] = None
                # print(f"'{letter}' accuracy: No data available")

        sorted_accuracy_data = {letter.lower(): accuracy_data[letter.lower()] for letter in letter_frequencies.keys()}

        bar_width = 0.1

        keys = list(sorted_accuracy_data.keys())
        values = list(sorted_accuracy_data.values())

        offset = bar_width * i - (len(model_names) - 1) * bar_width / 2  # Calculate the offset for each model
        x_positions_for_bars = [i + offset for i in range(len(sorted_accuracy_data))]

        plt.bar(x_positions_for_bars, values, width=bar_width, label=model_name)
        i += 1
        num_correct = 0
        letter_stats = {letter: {"wrong": 0, "total": 0} for letter in string.ascii_lowercase}

    fontsize = 16
    plt.xlim(-0.5, len(sorted_accuracy_data) - 0.5)
    plt.xticks([x - 0.37 for x in x_positions_for_bars], sorted_accuracy_data.keys(), fontsize=fontsize)
    plt.yticks([i * 10 for i in range(5)], fontsize=fontsize)
    plt.xlabel('Letters Sorted by Frequency', fontsize=fontsize)
    plt.ylabel('Percentage of Letters with Count Errors (%)', fontsize=fontsize)
    plt.legend(loc='best', fontsize=fontsize - 6)
    plt.show()


if __name__ == '__main__':
    main()
