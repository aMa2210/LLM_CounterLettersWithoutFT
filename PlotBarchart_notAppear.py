import csv
import matplotlib.pyplot as plt

model_names = ['LlaMa3.1-8B_IFCorrect', 'LlaMa3.1-70B_IFCorrect', 'LLama3.2-11B_IFCorrect', 'Mistral-7B_IFCorrect',
               'Mixtral-8x7b_IFCorrect', 'Gemma2-9B_IFCorrect', 'GPT4o-mini_IFCorrect', 'GPT4o_IFCorrect']
model_names = [name.replace('_IFCorrect', '') for name in model_names]

a = [514/9955,79/10000,670/9990,1348/9952,2765/9933,922/9982,187/9983,20/9976]
a = [i*100 for i in a]
fontsize = 16

plt.figure(figsize=(10, 6))

bars = plt.bar(model_names, a)

# 在每个柱的顶部添加数值
for bar in bars:
    height = bar.get_height()  # 获取每个柱的高度
    plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', ha='center', va='bottom', fontsize=fontsize)

# plt.title('Percentage of Words with Count Errors per Model', fontsize=fontsize)
plt.xlabel('Model Name', fontsize=fontsize)
plt.ylabel('Percentage of Answers with Non-Existent Letters (%)', fontsize=fontsize)
plt.xticks(fontsize=fontsize)

plt.ylim(0, 1)
plt.yticks([i * 10 for i in range(4)], fontsize=fontsize)
plt.tight_layout()
plt.show()