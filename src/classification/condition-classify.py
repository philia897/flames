from PIL import Image
import os
import torchvision.transforms as transforms
import torch
import json
from classification.utils import load_model
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def img_prediction(model, img_paths:list, mapping:dict):
    predictions_dict = {}

    # Make predictions for each image
    for image_filepath in tqdm(img_paths):
        # Load the image
        image = Image.open(image_filepath)

        # Preprocess the image
        input_tensor = transform(image).unsqueeze(0)

        # Make the prediction
        with torch.no_grad():
            output = model(input_tensor.to(device))
            _, predicted_class = torch.max(output, 1)
            predicted_class = predicted_class.item()

        # Store the prediction in the dictionary
        filename = os.path.basename(image_filepath)
        predictions_dict[filename] = mapping[predicted_class]
    
    return predictions_dict

def init_log_file(img_fns: list, output_file: str):
    with open(output_file, 'w') as f:
        imgs = []
        for img_fn in img_fns:
            imgs.append({
                "name": img_fn,
                "attributes": {}
            })
        json.dump(imgs, f, indent=4)

def save_pred_to_log(pred_dict:dict, attr_name:str, output_file:str):
    # Save the predictions to a JSON file
    with open(output_file, 'r+') as f:
        imgs = json.load(f)
        for item in imgs:
            item['attributes'][attr_name] = pred_dict[item['name']]
        f.seek(0)  # Move the file cursor to the beginning
        json.dump(imgs, f, indent=4)
        # Truncate the remaining contents (if any)
        f.truncate()

if __name__ == "__main__":
    reset = False
    img_folder = '/home/zekun/drivable/data/bdd100k/images/10k/train'
    output_json_file = "/home/zekun/drivable/data/bdd100k/labels/10k/bdd100k_labels_images_attributes_train.json"
    img_paths = [os.path.join(img_folder, filename) for filename in os.listdir(img_folder)]

    if reset:
        init_log_file([fn for fn in os.listdir(img_folder)], output_json_file)
        exit(0)

    model_logs_file = "/home/zekun/drivable/outputs/classification/logs/model_log.json"

    # model_file = "/home/zekun/drivable/outputs/classification/models/weather-20230720_092058.pth"
    # model_file = "/home/zekun/drivable/outputs/classification/models/scene-20230720_184731.pth"
    model_file = "/home/zekun/drivable/outputs/classification/models/timeofday-20230720_201618.pth"
    model_name = os.path.basename(model_file)
    
    with open(model_logs_file, 'r') as f:
        model_logs = json.load(f)
        model_info = model_logs[model_name]
        print("********** Model Info ***********")
        for k,v in model_info.items():
            print(k,v)
        print("*********************************")
    num_classes = len(model_info['mapping'])
    attr_name = model_info['condition_attr']
    idx_to_attr_mapping = {v: k for k,v in model_info['mapping'].items()}
    print(idx_to_attr_mapping)

    model = load_model(num_classes, model_file)
    model.eval().to(device)

    pred_rst = img_prediction(model, img_paths, idx_to_attr_mapping)
    for k,v in pred_rst.items():
        print(k,v)
        break

    save_pred_to_log(pred_rst, attr_name, output_json_file)

    print(f"Saved to {output_json_file}")

