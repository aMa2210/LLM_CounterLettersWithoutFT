import csv
import matplotlib.pyplot as plt
# from PlotLineChart import plot_lineChart
# model_name = 'LlaMa3.1-8B_IFCorrect'
# model_name = 'Gemma2-9B_IFCorrect'
# model_name = 'Mixtral-7B_IFCorrect'
# model_name = 'GPT4o_IFCorrect'
model_name = 'GPT4o-mini_IFCorrect'

file_frequency = 'Results/frequency_tokens_count_1w_list.csv'
file_words = 'Results/Random_10000_words.csv'
# file_words = 'Results/Top_10000_words.csv'
token_frequency = {}
with open(file_frequency, 'r', encoding='utf-8') as freq_file:
    reader = csv.DictReader(freq_file)
    for row in reader:
        token = row['Token']
        frequency = int(row['Frequency'])
        token_frequency[token] = frequency

word_total_frequency = {}  # 用于存储每个 word 对应的累加频率
with open(file_words, 'r', encoding='utf-8') as token_file:
    reader = csv.DictReader(token_file)
    for row in reader:
        word = row['Word']
        token_num = int(row['Tokens_Number'])
        tokens = row['Tokens'].split()
        accuracy = int(row[model_name])
        total_frequency = 0


        for token in tokens:
            if token in token_frequency:
                # total_frequency += token_frequency[token]
                if total_frequency == 0:
                    total_frequency = token_frequency[token]
                elif total_frequency > token_frequency[token]:
                    total_frequency = token_frequency[token]
            else:
                print(f'{token} does not exist in that file')


        word_total_frequency[word] = {'total_frequency': total_frequency, 'accuracy': accuracy}
        # word_total_frequency[word] = {'total_frequency': total_frequency / token_num, 'accuracy': accuracy}
sorted_word_total_frequency = sorted(word_total_frequency.items(), key=lambda item: item[1]['total_frequency'], reverse=True)

cumulative_accuracy = []
current_sum = 0
for word, values in sorted_word_total_frequency:
    current_sum += values['accuracy']
    cumulative_accuracy.append(current_sum)

plt.figure(figsize=(18, 6))

plt.plot(cumulative_accuracy, linewidth=6)
plt.xlabel(f'Sorted by Least Frequency Token')
plt.ylabel('Cumulative Number of Words with correct counts')
plt.title(f'Random words_{model_name.replace("_IFCorrect", "")}_Least Token Frequency')
plt.xticks(range(0, len(cumulative_accuracy), 1000))
plt.xlim(0, len(cumulative_accuracy))
plt.ylim(0, max(cumulative_accuracy))
plt.grid(False)
plt.show()


