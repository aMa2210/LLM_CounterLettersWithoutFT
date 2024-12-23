import csv
import matplotlib.pyplot as plt
from collections import defaultdict
from collections import Counter
import json

def main():
    model_names = ['LlaMa3.1-8B_IFCorrect', 'LlaMa3.1-70B_IFCorrect', 'LLama3.2-11B_IFCorrect', 'Mistral-7B_IFCorrect',
                   'Mixtral-8x7b_IFCorrect', 'Gemma2-9B_IFCorrect', 'GPT4o-mini_IFCorrect', 'GPT4o_IFCorrect']
    model_names_json = ['LlaMa3.1_8B', 'LlaMa3.1_70B', 'LLama3.2_11B', 'mistral_7B',
                   'mixtral-8x7b', 'gemma2-9B', 'gpt4o_mini', 'gpt4o']
    model_names_json = ['Results/' + result + '.json' for result in model_names_json]

    plot_barChart('Word', model_names, model_names_json)


def letter_count(word, json_word):

    count = {
        1: {'wrong': 0, 'total': 0},
        2: {'wrong': 0, 'total': 0},
        3: {'wrong': 0, 'total': 0},
        4: {'wrong': 0, 'total': 0},
        5: {'wrong': 0, 'total': 0},
        6: {'wrong': 0, 'total': 0},
        7: {'wrong': 0, 'total': 0},
        8: {'wrong': 0, 'total': 0},
    }
    all_chars = Counter(word)
    if isinstance(json_word, dict):
        for char in all_chars:
            count[all_chars[char]]['total'] +=1
            if char in json_word:
                if all_chars[char] != json_word[char]:
                    count[all_chars[char]]['wrong'] += 1

    # letter_counts = Counter(word)
    # return frequency of the most common letter
    return count


def plot_barChart(orderColumn, model_names, model_names_json):
    plt.figure(figsize=(18, 6))
    file_path = 'Results/Random_10000_words.csv'

    # Define the width of the bars
    bar_width = 0.1
    # Set the positions for the bars
    x_positions = range(len(model_names))

    for idx, (model_name, model_name_json) in enumerate(zip(model_names, model_names_json)):

        with open(model_name_json, "r") as json_file:  #load json format answer i.e letter-level
            data_json = json.load(json_file)
        for key, value in data_json.items():  # Parse the JSON-formatted value in file a from a string into a dictionary to compare
            try:
                data_json[key] = json.loads(value)
            except:
                # print('errorKey: ' + key)
                pass

        data = []
        count = {
            1: {'wrong': 0, 'total': 0},
            2: {'wrong': 0, 'total': 0},
            3: {'wrong': 0, 'total': 0},
            4: {'wrong': 0, 'total': 0},
            5: {'wrong': 0, 'total': 0},
            6: {'wrong': 0, 'total': 0},
            7: {'wrong': 0, 'total': 0},
            8: {'wrong': 0, 'total': 0},
        }
        with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                dict1 = letter_count(row[orderColumn], data_json[row[orderColumn]])
                for key, sub_dict in dict1.items():
                    for sub_key, value in sub_dict.items():
                        count[key][sub_key] += value

        # grouped_data = defaultdict(int)
        # grouped_count = defaultdict(int)
        # for item in data:
        #     first_value = item[0]  # Letters_Number
        #     second_value = 1 - item[1]  # error
        #     grouped_data[first_value] += second_value
        #     grouped_count[first_value] += 1
        #
        # grouped_data = {key: grouped_data[key] for key in grouped_data if
        #                 1 <= key <= 4}  # Only retain the data within this range, as the other data samples are too few.
        # grouped_count = {key: grouped_count[key] for key in grouped_count if 1 <= key <= 4}
        # print(grouped_data)
        # print(grouped_count)
        # print('********************')
        count = {key: count[key] for key in count if 1 <= key <= 4}
        grouped_average = {key: (count[key]['wrong'] / count[key]['total']) * 100 for key in count}

        keys = list(grouped_average.keys())
        values = list(grouped_average.values())

        # Adjust the positions for each model's bars
        offset = bar_width * idx - (len(model_names) - 1) * bar_width / 2  # Calculate the offset for each model
        plt.bar([key + offset for key in keys], values, width=bar_width, label=model_name)
    # for key, value in sorted(grouped_count.items()):
    #     print(f'{key}: {value}')
    fontsize = 16
    # plt.xticks(range(3, 15), fontsize=fontsize)
    # plt.xticks(range(1, 8), fontsize=fontsize)
    plt.xticks(range(1, 5), fontsize=fontsize)
    plt.yticks([i * 10 for i in range(11)], fontsize=fontsize)
    plt.xlabel('Letter Multiplicity', fontsize=fontsize)
    plt.ylabel('Percentage of Letters with Count Errors (%)', fontsize=fontsize)
    plt.legend(loc='upper left', fontsize=fontsize)
    plt.show()


if __name__ == '__main__':
    main()
