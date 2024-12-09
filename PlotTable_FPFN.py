import json
import pandas as pd
from collections import Counter
from openpyxl.styles import Font

def main():
    model_names = ['LlaMa3.1-8B_IFCorrect', 'LlaMa3.1-70B_IFCorrect', 'LLama3.2-11B_IFCorrect', 'Mistral-7B_IFCorrect',
                   'Mixtral-8x7b_IFCorrect', 'Gemma2-9B_IFCorrect', 'GPT4o-mini_IFCorrect', 'GPT4o_IFCorrect']
    model_names_json = ['LlaMa3.1_8B', 'LlaMa3.1_70B', 'LLama3.2_11B', 'mistral_7B',
                        'mixtral-8x7b', 'gemma2-9B', 'gpt4o_mini', 'gpt4o']
    model_names_json = ['Results/' + result + '.json' for result in model_names_json]

    output_file = 'Results/Real_Pred_Table.xlsx'
    save_to_excel(output_file, model_names, model_names_json)


def letter_count(word, json_word):
    count = {i: {j: 0 for j in range(9)} for i in range(1, 9)}
    count_not_exist = {
        'non_existent': 0,
        'total': 0
    }
    all_chars = Counter(word)
    if isinstance(json_word, dict):
        for char in all_chars:
            if char in json_word:
                count[all_chars[char]][json_word[char]] += 1
                # if all_chars[char] == 8:
                #     print(word)
            else:
                count[all_chars[char]][0] += 1

    # count number of times that the model generate an answer including a letter while it actually doesn't exist in the word
    if isinstance(json_word, dict):
        count_not_exist['total'] +=1
        if any(char not in all_chars for char in json_word):
            count_not_exist['non_existent'] += 1


    return count, count_not_exist


def save_to_excel(file_path, model_names, model_names_json):
    """
    Save the count data for each model to separate sheets in an Excel file.
    """
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        for model_name, model_name_json in zip(model_names, model_names_json):
            with open(model_name_json, "r") as json_file:  # Load JSON format answers
                data_json = json.load(json_file)
            for key, value in data_json.items():  # Parse the JSON-formatted value
                try:
                    data_json[key] = json.loads(value)
                except:
                    pass

            count = {i: {j: 0 for j in range(9)} for i in range(1, 9)}
            count_not_exist = {
                'non_existent': 0,
                'total': 0
            }

            for word, json_word in data_json.items():
                word_count, word_count_nonExist = letter_count(word, json_word)
                for real_count in word_count:
                    for pred_count in word_count[real_count]:
                        count[real_count][pred_count] += word_count[real_count][pred_count]
                for entity in count_not_exist:
                    count_not_exist[entity] += word_count_nonExist[entity]

            # Convert the count dictionary to a pandas DataFrame
            df = pd.DataFrame.from_dict(count, orient='index')
            df.index.name = 'Reality\Prediction'
            df.columns = [f'{i}' for i in range(9)]

            # Add a 'Total' column
            df['Total'] = df.sum(axis=1)

            # Format the data with percentages
            df_formatted = df.apply(lambda row: [
                f"{int(value)} ({value / row['Total'] * 100:.0f}%)" if row['Total'] > 0 else f"{int(value)} (0%)"
                for value in row[:-1]
            ] + [f"{int(row['Total'])} (100%)"], axis=1)

            # Convert back to DataFrame for Excel
            df_formatted = pd.DataFrame(df_formatted.tolist(), columns=[f'{i}' for i in range(9)] + ['Total'], index=df.index)

            print(count_not_exist)

            # Write to a sheet named after the model
            df_formatted.to_excel(writer, sheet_name=model_name.replace('_IFCorrect','')[:31])  # Sheet name limited to 31 characters

            workbook = writer.book
            sheet = workbook[model_name.replace('_IFCorrect','')[:31]]

            # Set the font of all cells to Times New Roman
            for row in sheet.iter_rows():
                for cell in row:
                    cell.font = Font(name='Times New Roman', size=12)


            print(f"Saved data for {model_name} to sheet.")

    print(f"All data saved to {file_path}")


if __name__ == "__main__":
    main()
