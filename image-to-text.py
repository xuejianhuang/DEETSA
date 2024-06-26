import json
import os.path
from PIL import Image
import time
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration


def img_to_text(dir_root, json_file, out_file):
    """
    Convert images to text captions and save the results in a JSON file.

    Parameters:
    dir_root (str): Root directory containing the images and JSON file.
    json_file (str): Name of the JSON file containing image paths.
    out_file (str): Name of the output JSON file to save the results.

    Returns:
    None
    """
    start_t = time.time()

    # Initialize the processor and model
    processor = BlipProcessor.from_pretrained("./blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("./blip-image-captioning-large").to("cuda")

    # Read the contents of the JSON file
    with open(os.path.join(dir_root, json_file), 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Process each image and generate corresponding text
    for key, value in tqdm(data.items()):
        img_path = os.path.join(dir_root, value['image_path'])
        raw_image = Image.open(img_path).convert('RGB')

        inputs = processor(raw_image, return_tensors="pt").to("cuda")
        out = model.generate(**inputs)

        # Add the generated text to the JSON data
        value['img-to-text_en'] = processor.decode(out[0], skip_special_tokens=True)

    # Write the updated data back to a JSON file
    with open(os.path.join(dir_root, out_file), 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

    end_t = time.time()
    print("Time taken:", end_t - start_t)


if __name__ == '__main__':
    dir_root = '../data/Weibo/'
    json_file = 'dataset_items_merged.json'
    out_file = 'output.json'
    img_to_text(dir_root, json_file, out_file)
