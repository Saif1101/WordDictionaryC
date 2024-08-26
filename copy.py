import torch
import torchvision
from PIL import Image
import torchvision.transforms as transforms

# Define image preprocessing
def preprocess_image(image_path, img_size=640):
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    transform_pipeline = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = transform_pipeline(image).unsqueeze(0)
    return image, original_size

# Load the model
def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_path, map_location=device)
    model.eval()
    return model, device

# Non-Maximum Suppression and Bounding Box Processing
def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, max_det=300):
    nc = prediction.shape[2] - 5
    xc = prediction[..., 4] > conf_thres
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]
        if not x.shape[0]:
            continue
        x[:, 5:] *= x[:, 4:5]
        box = xywh2xyxy(x[:, :4])
        conf, j = x[:, 5:].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
        if not x.shape[0]:
            continue
        n = x.shape[0]
        c = x[:, 5:6] * (0 if nc == 1 else 4096)
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:
            i = i[:max_det]
        output[xi] = x[i]
    return output

def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

# Main Detection Function
def detect(model, device, image_path):
    image, original_size = preprocess_image(image_path)
    image = image.to(device)
    with torch.no_grad():
        prediction = model(image)
        detections = non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45)
    return detections, original_size

# Load model
model_path = 'path_to_your_model.pt'
model, device = load_model(model_path)

# Run detection
image_path = 'path_to_your_image.jpg'
detections, original_size = detect(model, device, image_path)

# Display the detections
print("Detected objects:")
for det in detections:
    if det is not None and len(det):
        for *xyxy, conf, cls in reversed(det):
            print(f"Class: {cls.item()}, Confidence: {conf.item()}, BBox: {xyxy}")
