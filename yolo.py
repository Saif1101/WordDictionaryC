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
    img = np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.0  # Normalization added here
    img = torch.tensor(img, dtype=torch.float32)
    img = img.unsqueeze(0)  # Add batch dimension
    return img

# Non-Maximum Suppression to filter predictions
def non_max_suppression(predictions, conf_thresh=0.5, iou_thresh=0.5):
    boxes = []
    for pred in predictions:
        # Filter out low confidence scores
        pred = pred[pred[:, 4] > conf_thresh]
        if not pred.size(0):
            continue
        
        # Convert (x_center, y_center, width, height) to (x1, y1, x2, y2)
        pred[:, :4] = xywh_to_xyxy(pred[:, :4])
        
        # Sort by confidence score
        pred = pred[(-pred[:, 4]).argsort()]
        
        # Apply NMS
        keep_boxes = []
        while pred.size(0):
            best_box = pred[0]
            keep_boxes.append(best_box)
            if pred.size(0) == 1:
                break
            ious = bbox_iou(best_box.unsqueeze(0), pred[1:])
            pred = pred[1:][ious < iou_thresh]
        boxes.append(torch.stack(keep_boxes))
    return boxes

# Function to convert (x_center, y_center, width, height) to (x1, y1, x2, y2)
def xywh_to_xyxy(box):
    xyxy_box = torch.zeros_like(box)
    xyxy_box[:, 0] = box[:, 0] - box[:, 2] / 2  # x1
    xyxy_box[:, 1] = box[:, 1] - box[:, 3] / 2  # y1
    xyxy_box[:, 2] = box[:, 0] + box[:, 2] / 2  # x2
    xyxy_box[:, 3] = box[:, 1] + box[:, 3] / 2  # y2
    return xyxy_box

# Calculate IOU for NMS
def bbox_iou(box1, box2):
    # Calculate the intersection
    inter_rect_x1 = torch.max(box1[:, 0], box2[:, 0])
    inter_rect_y1 = torch.max(box1[:, 1], box2[:, 1])
    inter_rect_x2 = torch.min(box1[:, 2], box2[:, 2])
    inter_rect_y2 = torch.min(box1[:, 3], box2[:, 3])
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1, min=0)
    
    # Calculate the union
    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area

# Inference
def infer(image_path):
    input_size = 640  # Adjust this according to your model input size
    img = preprocess_image(image_path, input_size)
    with torch.no_grad():
        predictions = model(img)[0]  # Assuming model returns (batch_size, num_boxes, 6)
        predictions = non_max_suppression([predictions], conf_thresh=0.5, iou_thresh=0.5)
    return predictions

# Example usage
image_path = 'path_to_your_image.jpg'
predictions = infer(image_path)
print(predictions)
