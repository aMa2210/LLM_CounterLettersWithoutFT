import csv
import matplotlib.pyplot as plt
# from PlotLineChart import plot_lineChart
# model_name = 'LlaMa3.1-8B_IFCorrect'
# model_name = 'Gemma2-9B_IFCorrect'
# model_name = 'Mixtral-7B_IFCorrect'
# model_name = 'GPT4o_IFCorrect'
model_name = 'GPT4o-mini_IFCorrect'

file_frequency = 'Results/frequency_tokens_count_1w_list.csv'
# file_words = 'Results/Random_10000_words.csv'
file_words = 'Results/Top_10000_words.csv'
token_frequency = {}
with open(file_frequency, 'r', encoding='utf-8') as freq_file:
    reader = csv.DictReader(freq_file)
    for row in reader:
        token = row['Token']
        frequency = int(row['Frequency'])
        token_frequency[token] = frequency

token_stats = {}  # Key: token, Value: {'correct_count': x, 'total_count': y}

with open(file_words, 'r', encoding='utf-8') as token_file:
    reader = csv.DictReader(token_file)
    for row in reader:
        word = row['Word']
        tokens = row['Tokens'].split()  # Split tokens
        accuracy = int(row[model_name])  # Get the accuracy for the word

        for token in tokens:
            if token not in token_stats:
                token_stats[token] = {'correct_count': 0, 'wrong_count': 0, 'total_count': 0}

            # Increment the total count for this token
            token_stats[token]['total_count'] += 1

            # If the word has accuracy 0, it's an error for that token
            if accuracy == 0:
                token_stats[token]['wrong_count'] += 1  # Count the errors
            else:
                token_stats[token]['correct_count'] += 1  # Count the correct instances

# Step 3: Prepare data for output
output_data = []

for token, counts in token_stats.items():
    wrong_count = counts['wrong_count']
    total_count = counts['total_count']
    error_rate = wrong_count / total_count if total_count > 0 else 0  # Avoid division by zero
    frequency = token_frequency.get(token, 0)  # Get frequency from the frequency file, default 0 if not found

    # Format 'Correct/Wrong' column as 'correct_count/total_count'
    wrong_total = f'{wrong_count} of {total_count}'

    # Append the formatted data
    output_data.append({
        'Token': token,
        'Error_Rate': error_rate,
        'Frequency': frequency,
        'Wrong/Total': wrong_total
    })

# Step 4: Sort by Error Rate (descending order)
output_data = sorted(output_data, key=lambda x: x['Error_Rate'], reverse=True)

# Step 5: Write the results to a CSV file
output_file = f'Results/{model_name.replace("_IFCorrect", "")}_token_error_rates.csv'
with open(output_file, 'w', encoding='utf-8', newline='') as out_file:
    fieldnames = ['Token', 'Error_Rate', 'Frequency', 'Wrong/Total']
    writer = csv.DictWriter(out_file, fieldnames=fieldnames)

    writer.writeheader()
    for data in output_data:
        writer.writerow(data)

print(f'Results have been written to {output_file}')


