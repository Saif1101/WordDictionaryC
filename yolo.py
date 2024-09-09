import torch
import cv2
import numpy as np

# Load the model
model_path = 'path_to_your_model.pt'
model = torch.load(model_path)
model = model['model'].float()
model.eval()

# Function to preprocess the image
def preprocess_image(image_path, input_size):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (input_size, input_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.0
    img = torch.tensor(img, dtype=torch.float32)
    img = img.unsqueeze(0)
    return img

# Non-Maximum Suppression to filter predictions
def non_max_suppression(prediction, conf_thresh=0.5, iou_thresh=0.5):
    # Filter out predictions with low confidence
    conf_mask = (prediction[:, :, 4] > conf_thresh).squeeze()
    prediction = prediction[conf_mask]
    
    # Perform NMS
    keep_boxes = []
    while prediction.size(0):
        _, indices = prediction[:, 4].sort(descending=True)
        best_box = prediction[indices[0]]
        
        keep_boxes += [best_box]
        if prediction.size(0) == 1:
            break
        
        ious = bbox_iou(best_box.unsqueeze(0), prediction[indices[1:]], x1y1x2y2=False)
        prediction = prediction[indices[1:]][ious < iou_thresh]
        
    if keep_boxes:
        return torch.stack(keep_boxes)
    return torch.tensor([])

# Calculate IOU for NMS
def bbox_iou(box1, box2, x1y1x2y2=True):
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Coordinates are already provided as x1, y1, x2, y2
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # Intersection area
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1, min=0)

    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area + b2_area - inter_area

    return inter_area / union_area

# Inference
def infer(image_path):
    input_size = 640
    img = preprocess_image(image_path, input_size)
    with torch.no_grad():
        predictions = model(img)
        filtered_predictions = non_max_suppression(predictions, conf_thresh=0.5, iou_thresh=0.5)
    return filtered_predictions

# Example usage
image_path = 'path_to_your_image.jpg'
predictions = infer(image_path)
print(predictions)
