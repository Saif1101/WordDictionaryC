import torch
import torchvision
import numpy as np
import cv2

class YOLOv8:
    def __init__(self, model_path, device='', img_size=640, conf_thres=0.25, iou_thres=0.45):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path)
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = self.model.names if hasattr(self.model, 'names') else None
        self.max_det = 300

    def load_model(self, model_path):
        model = torch.load(model_path, map_location=self.device)['model'].float().fuse().eval()
        return model

    def preprocess(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img, _, _ = letterbox(img, new_shape=self.img_size)
        img = img.transpose((2, 0, 1))  # HWC to CHW
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    def postprocess(self, output, img, orig_img):
        print(f"Output type: {type(output)}")
        if isinstance(output, tuple):
            print(f"Output tuple length: {len(output)}")
            for i, item in enumerate(output):
                print(f"Item {i} type: {type(item)}, shape: {item.shape if hasattr(item, 'shape') else 'N/A'}")
            
            # Assume the first element of the tuple contains the detections
            pred = output[0]
        else:
            pred = output

        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, max_det=self.max_det)
        results = []
        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orig_img.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    results.append((xyxy, conf, cls))
        return results

    def predict(self, img_path):
        orig_img = cv2.imread(img_path)
        img = self.preprocess(img_path)
        
        with torch.no_grad():
            output = self.model(img)
        
        results = self.postprocess(output, img, orig_img)
        
        return [
            {
                'bbox': [int(x.item()) for x in xyxy],
                'class': self.classes[int(cls)] if self.classes else int(cls),
                'confidence': float(conf)
            } for xyxy, conf, cls in results
        ]

# The helper functions (letterbox, non_max_suppression, xywh2xyxy, scale_coords, clip_coords) remain the same

# Usage example
if __name__ == "__main__":
    model_path = "path/to/your/yolov8n.pt"
    image_path = "path/to/your/image.png"

    try:
        yolo = YOLOv8(model_path)
        results = yolo.predict(image_path)

        for detection in results:
            print(f"Detected {detection['class']} with confidence {detection['confidence']:.2f} at {detection['bbox']}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
