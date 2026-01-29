import os
import torch
from ultralytics import YOLO


def is_correct(res, expected_result):
    if expected_result:
        return torch.any(res.boxes.cls == 1).item()
    else:
        return torch.all(res.boxes.cls == 0).item()


img_path = 'hockey-data/images/val'
label_path = 'hockey-data/labels/val'
model = YOLO('violence-detect-v2.pt')

num_right, num_wrong = 0, 0
for img_file in os.listdir(img_path):
    filename = f'{img_path}/{img_file}'
    results = model.predict(filename)
    for res in results:
        if is_correct(res, img_file.startswith('fi')):
            num_right += 1
        else:
            for box, cls in zip(res.boxes.xyxy, res.boxes.cls):
                print(filename, box, cls.item())
            num_wrong += 1

print(f'Correct: {num_right}, Wrong: {num_wrong}')
