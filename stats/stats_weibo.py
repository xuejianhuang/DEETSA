import json
import numpy as np
import jieba

if __name__ == '__main__':

    # Load the JSON file
    with open('../../data/Weibo/dataset_items_merged.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract all captions with label == 1
    # captions = [item['caption'] for item in data.values() if item['label'] == 1]
    captions = [item['caption'] for item in data.values()]

    # Calculate the length of each caption (using jieba for Chinese word segmentation)
    caption_lengths = [len(list(jieba.cut(caption))) for caption in captions]

    # Count the number of samples with a length greater than 40
    count_greater_than_40 = sum(length > 40 for length in caption_lengths)

    # Count the number of samples with a length no greater than 40
    count_greater_less_40 = sum(length <= 40 for length in caption_lengths)

    # Compute the average length, maximum length, minimum length, and standard deviation
    average_length = np.mean(caption_lengths)
    max_length = np.max(caption_lengths)
    min_length = np.min(caption_lengths)
    std_dev = np.std(caption_lengths)

    # Output the results
    print(f"Average length of weibo dataset: {average_length:.2f}")
    print(f"Maximum length of weibo dataset: {max_length}")
    print(f"Minimum length of weibo dataset: {min_length}")
    print(f"Standard Deviation of weibo dataset: {std_dev:.2f}")

    print(f"Number of samples with length greater than 40 in the weibo dataset: {count_greater_than_40}")

    print(f"The number of samples with length no greater than 40 in the weibo dataset: {count_greater_less_40}")
