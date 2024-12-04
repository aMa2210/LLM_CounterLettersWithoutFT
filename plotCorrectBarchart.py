import csv
import matplotlib.pyplot as plt

row_name = ['LlaMa3.1-8B_IFCorrect', 'LlaMa3.1-70B_IFCorrect', 'LLama3.2-11B_IFCorrect',  'Mistral-7B_IFCorrect', 'Mixtral-8x7b_IFCorrect','Gemma2-9B_IFCorrect','GPT4o-mini_IFCorrect', 'GPT4o_IFCorrect']

column_sums = {name: 0 for name in row_name}

# 读取CSV文件
with open('Results/Random_10000_words.csv', mode='r', newline='', encoding='utf-8') as file:
    reader = csv.DictReader(file, delimiter=',')

    for row in reader:
        for name in row_name:
            try:
                column_sums[name] += float(row[name])
            except ValueError:
                raise ValueError(f"Invalid value encountered for {name}")

fontsize = 16

plt.figure(figsize=(10, 6))
modified_keys = [key.replace('_IFCorrect', '') for key in column_sums.keys()]  # 替换模型名称中的 '_IFCorrect'
modified_values = [(10000-value) / 10000 for value in column_sums.values()]  # 计算百分比

bars = plt.bar(modified_keys, modified_values)

# 在每个柱的顶部添加数值
for bar in bars:
    height = bar.get_height()  # 获取每个柱的高度
    plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', ha='center', va='bottom', fontsize=fontsize)

# plt.title('Percentage of Words with Count Errors per Model', fontsize=fontsize)
plt.xlabel('Model Name', fontsize=fontsize)
plt.ylabel('Percentage of Words with Count Errors', fontsize=fontsize)
plt.xticks(fontsize=fontsize)

plt.ylim(0, 1)
plt.yticks([i * 0.1 for i in range(11)], fontsize=fontsize)
plt.tight_layout()
plt.show()
