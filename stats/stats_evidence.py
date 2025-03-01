import json
import os

base_path = "../../data/Twitter/"

def calculate_lengths_from_direct_annotation(direct_path):
    """
    Read direct_annotation.json from the direct_path directory and calculate the sum of the specified array lengths.
    :param direct_path: Path to the direct_path directory
    :return: Sum of array lengths
    """
    direct_annotation_path = os.path.join(direct_path, 'direct_annotation.json')
    if os.path.exists(direct_annotation_path):
        with open(direct_annotation_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Calculate the sum of lengths for images_with_captions, images_with_no_captions, images_with_caption_matched_tags
        total_length = len(data.get('images_with_captions', [])) + \
                       len(data.get('images_with_no_captions', [])) + \
                       len(data.get('images_with_caption_matched_tags', []))
        return total_length
    return 0


def calculate_lengths_from_inverse_annotation(inv_path):
    """
    Read inverse_annotation.json from the inv_path directory and calculate the sum of the specified array lengths.
    :param inv_path: Path to the inv_path directory
    :return: Sum of array lengths
    """
    inverse_annotation_path = os.path.join(inv_path, 'inverse_annotation.json')
    if os.path.exists(inverse_annotation_path):
        with open(inverse_annotation_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Calculate the sum of lengths for all_fully_matched_captions, all_partially_matched_captions,
        # partially_matched_no_text, fully_matched_no_text
        total_length = len(data.get('all_fully_matched_captions', [])) + \
                       len(data.get('all_partially_matched_captions', [])) + \
                       len(data.get('partially_matched_no_text', [])) + \
                       len(data.get('fully_matched_no_text', []))
        return total_length
    return 0


def calculate_statistics(lengths):
    """
    Compute the maximum, minimum, and average values of an array of lengths.
    :param lengths: Array of lengths
    :return: Maximum value, minimum value, and average value
    """
    if lengths:
        max_length = max(lengths)
        min_length = min(lengths)
        avg_length = sum(lengths) / len(lengths)
        return max_length, min_length, avg_length
    return 0, 0, 0


def main():
    dataset_file = base_path + 'dataset_items_merged.json'

    # Read the dataset_items_merged.json file
    with open(dataset_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    # Store the lengths of direct_length and inv_length separately
    direct_lengths = []
    inv_lengths = []

    # Iterate over each data item
    items = [item for item in dataset.values() if item['label'] == 0]
    # items = [item for item in dataset.values()]

    for item in items:
        direct_path = base_path + item.get('direct_path')
        inv_path = base_path + item.get('inv_path')

        # Compute length from direct_path
        direct_length = calculate_lengths_from_direct_annotation(direct_path)
        direct_lengths.append(direct_length)

        # Compute length from inv_path
        inv_length = calculate_lengths_from_inverse_annotation(inv_path)
        inv_lengths.append(inv_length)

    # Compute the maximum, minimum, and average values of direct_length and inv_length
    direct_max, direct_min, direct_avg = calculate_statistics(direct_lengths)
    inv_max, inv_min, inv_avg = calculate_statistics(inv_lengths)

    # Output statistical results
    print(f"Textual Evidence - Maximum Total Length: {inv_max}, Minimum Total Length: {inv_min}, Average Total Length: {inv_avg:.2f}")
    print(f"Visual Evidence - Maximum Total Length: {direct_max}, Minimum Total Length: {direct_min}, Average Total Length: {direct_avg:.2f}")


if __name__ == '__main__':
    main()
