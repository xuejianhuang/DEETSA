import json
import jieba
from wordcloud import WordCloud
import matplotlib.pyplot as plt


def generate_wordcloud(json_data, font_path, output_path, lang='zh'):
    """
    Generate a word cloud based on the given JSON data and font path.

    Parameters:
    json_data (str): Path to the JSON file containing the data.
    font_path (str): Path to the font file to ensure correct display of Chinese or other languages.
    output_path (str): Path to save the generated word cloud image.
    lang (str): Language of the text data ('zh' for Chinese, defaults to 'zh').

    Returns:
    None
    """
    # Read the original JSON file
    with open(json_data, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract all caption content
    captions = " ".join(item["caption"].lower() for item in data.values())

    if lang == 'zh':
        # Perform segmentation for Chinese text
        words = jieba.lcut(captions)
    else:
        # For other languages, add segmentation methods as needed
        words = captions.split()  # Simple split by spaces

    # Load the stop words list
    stopwords = set()
    with open('stopwords.txt', 'r', encoding='utf-8') as f:
        for line in f:
            stopwords.add(line.strip())

    # Remove stop words and count word frequencies
    word_freq = {}
    for word in words:
        word = word.strip()
        if word not in stopwords:
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1

    # Generate the word cloud
    wordcloud = WordCloud(width=1000, height=1000, background_color='white', font_path=font_path,
                          max_words=100).generate_from_frequencies(word_freq)
    wordcloud.to_file(output_path)  # Save the word cloud image

    # Display the word cloud
    plt.figure(figsize=(20, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    # json_data='../data/weibo/dataset_items_merged.json'
    # font_path = 'C:/Windows/Fonts/simhei.ttf'  # Chinese font
    # output_path='wordcloud_weibo.png'
    json_data = '../data/Twitter/dataset_items_merged.json'
    font_path = 'C:\\Windows\\Fonts\\Arial.TTF'  # English font
    output_path = 'wordcloud_twitter.png'

    generate_wordcloud(json_data, font_path, output_path)
