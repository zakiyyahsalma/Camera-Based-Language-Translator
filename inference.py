import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from doctr.models import recognition
from doctr.transforms import Resize
from doctr.datasets import VOCABS
from googletrans import Translator  
from doctr.io import DocumentFile
from doctr.models import detection_predictor
from doctr.utils.geometry import detach_scores
import cv2
from copy import deepcopy

def apply_offset_to_boxes(boxes, offset_x_ratio=0.05, offset_y_ratio=0.1):
    new_boxes = []
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        box_w = xmax - xmin
        box_h = ymax - ymin
        pad_x = offset_x_ratio * box_w
        pad_y = offset_y_ratio * box_h

        xmin_p = max(0.0, xmin - pad_x)
        ymin_p = max(0.0, ymin - pad_y)
        xmax_p = min(1.0, xmax + pad_x)
        ymax_p = min(1.0, ymax + pad_y)

        new_boxes.append([xmin_p, ymin_p, xmax_p, ymax_p])
    return np.array(new_boxes, dtype=np.float32)



def crop_word_images(image_np, boxes, offset_x_ratio=0.05, offset_y_ratio=0.1):
    h, w = image_np.shape[:2]
    crops = []

    for box in boxes:
        xmin, ymin, xmax, ymax = (box * [w, h, w, h]).astype(int)

        box_w = xmax - xmin
        box_h = ymax - ymin
        pad_x = int(offset_x_ratio * box_w)
        pad_y = int(offset_y_ratio * box_h)

        xmin_p = max(0, xmin - pad_x)
        ymin_p = max(0, ymin - pad_y)
        xmax_p = min(w, xmax + pad_x)
        ymax_p = min(h, ymax + pad_y)

        cropped = image_np[ymin_p:ymax_p, xmin_p:xmax_p]
        crops.append(cropped)

    return crops

def sort_boxes(boxes, line_tol=0.03):

    boxes = np.array(boxes)

    y_centers = (boxes[:, 1] + boxes[:, 3]) / 2

    sorted_indices = np.argsort(y_centers)
    boxes = boxes[sorted_indices]

    lines = []
    current_line = [boxes[0]]
    for box in boxes[1:]:
        if abs((box[1] + box[3]) / 2 - (current_line[-1][1] + current_line[-1][3]) / 2) > line_tol:
            lines.append(current_line)
            current_line = [box]
        else:
            current_line.append(box)
    lines.append(current_line)

    sorted_boxes = [sorted(line, key=lambda b: b[0]) for line in lines]

    return [box for line in sorted_boxes for box in line]


def load_model(model_path, arch, vocab, input_size):
    model = recognition.__dict__[arch](
        pretrained=False,
        input_shape=(3, input_size, 4 * input_size),
        vocab=VOCABS[vocab],
    ).eval()
    checkpoint = torch.load(model_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.load_state_dict(checkpoint)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model

def preprocess_image(image_path, input_size):
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((input_size, 4 * input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    img_tensor = transform(img).unsqueeze(0)
    if torch.cuda.is_available():
        img_tensor = img_tensor.cuda()
    return img, img_tensor

def predict(model, image_tensor):
    with torch.inference_mode():
        output = model(image_tensor, return_preds=True)
    if len(output["preds"]):
        words, _ = zip(*output["preds"])
        return words[0]
    return "No prediction"

def translate_text(text, dest_lang="id"):
    translator = Translator()
    translated = translator.translate(text, dest=dest_lang)
    return translated.text

def visualize_all_in_one(image_np, boxes, prediction, translation, thickness=1):

    img = deepcopy(image_np)
    h, w = img.shape[:2]

    abs_boxes = deepcopy(boxes)
    abs_boxes[:, [0, 2]] *= w
    abs_boxes[:, [1, 3]] *= h
    abs_boxes = abs_boxes.astype(int)

    for box in abs_boxes:
        xmin, ymin, xmax, ymax = box
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 8))
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.title(f"Predicted: {prediction}\nTranslated: {translation}", fontsize=12, color='green', loc='left')
    plt.tight_layout()
    plt.show()



def main(args):
    model = load_model(args.model_path, args.arch, args.vocab, args.input_size)
    
    image_pil = Image.open(args.image_path).convert("RGB")
    image_np = np.array(image_pil)

    doc = DocumentFile.from_images(args.image_path)
    detector = detection_predictor(pretrained=True)
    detection_result = detector(doc)
    boxes_list, _ = detach_scores([detection_result[0]["words"]])
    boxes = boxes_list[0]

    visual_boxes = apply_offset_to_boxes(boxes, offset_x_ratio=0.04, offset_y_ratio=0.4)

    boxes = sort_boxes(boxes)
    cropped_images = crop_word_images(image_np, boxes, offset_x_ratio=0.04, offset_y_ratio=0.4)

    full_text = ""
    for crop in cropped_images:
        crop_pil = Image.fromarray(crop)
        crop_tensor = transforms.Compose([
            transforms.Resize((args.input_size, 4 * args.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])(crop_pil).unsqueeze(0)
        if torch.cuda.is_available():
            crop_tensor = crop_tensor.cuda()
        pred_text = predict(model, crop_tensor)
        full_text += pred_text + " "

    print(f"Predicted full text: {full_text.strip()}")
    translated = translate_text(full_text.strip())
    print(f"Translated to Indonesian: {translated}")

    visualize_all_in_one(image_np, visual_boxes, full_text.strip(), translated, thickness=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for text recognition model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the image file")
    parser.add_argument("--arch", type=str, required=True, help="Model architecture used for training")
    parser.add_argument("--vocab", type=str, required=True, help="Vocabulary used during training")
    parser.add_argument("--input_size", type=int, default=32, help="Input height for the model, width is 4x height")
    
    args = parser.parse_args()
    main(args)
