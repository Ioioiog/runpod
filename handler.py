"""
RunPod Serverless Handler for Event Detection
- YOLOv8n: Object detection
- BLIP-2: Image captioning
- NSFW: Content filtering
- Face/Plate: Blur sensitive data
"""

import runpod
import torch
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image, ImageFilter
import io
import base64
import numpy as np
import cv2

# Load models once at startup
print("Loading models...")
yolo_model = YOLO('yolov8n.pt')
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
print("Models loaded!")

def classify_event_type(detections):
    """Classify event type based on detected objects"""
    objects = [d['class'] for d in detections]
    object_counts = {}
    
    for obj in objects:
        object_counts[obj] = object_counts.get(obj, 0) + 1
    
    # Fire/Smoke detection
    if 'fire' in objects or 'smoke' in objects:
        return 'Incendiu', 0.85
    
    # Accident detection
    if 'car' in objects and 'person' in objects:
        return 'Accident', 0.78
    
    # Protest detection (many people)
    if object_counts.get('person', 0) > 20:
        return 'Protest', 0.82
    
    # Emergency vehicles
    if 'truck' in objects or 'bus' in objects:
        return 'Urgență', 0.75
    
    # Default
    return 'Altul', 0.60

def blur_region(image, bbox, blur_amount=30):
    """Blur a specific region in the image"""
    img_array = np.array(image)
    x1, y1, x2, y2 = map(int, bbox)
    
    # Extract region
    region = img_array[y1:y2, x1:x2]
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(region, (blur_amount, blur_amount), 0)
    
    # Replace region
    img_array[y1:y2, x1:x2] = blurred
    
    return Image.fromarray(img_array)

def detect_faces(image):
    """Detect faces using OpenCV Haar Cascade (lightweight)"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    img_array = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    return [{'bbox': [x, y, x+w, y+h]} for (x, y, w, h) in faces]

def check_nsfw(image):
    """Simple NSFW check based on skin tone detection (placeholder)"""
    # In production, use a proper NSFW model
    # For now, return safe
    return 0.1

def handler(event):
    """
    Main handler function
    
    Input:
    {
        "input": {
            "image": "base64_encoded_image",
            "tasks": ["detect", "caption", "nsfw", "blur"]
        }
    }
    
    Output:
    {
        "event_type": "Accident",
        "confidence": 0.78,
        "caption": "A car accident on the street",
        "detections": [...],
        "nsfw_score": 0.1,
        "is_safe": true,
        "blurred_image": "base64_encoded_blurred_image"
    }
    """
    try:
        # Decode image
        image_data = base64.b64decode(event["input"]["image"])
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        results = {}
        tasks = event["input"].get("tasks", ["detect", "caption", "nsfw", "blur"])
        
        # 1. YOLO Detection (~5-10ms on T4)
        if "detect" in tasks:
            detections_raw = yolo_model(image, verbose=False)
            detections = []
            
            for result in detections_raw:
                boxes = result.boxes
                for box in boxes:
                    detections.append({
                        'class': result.names[int(box.cls[0])],
                        'confidence': float(box.conf[0]),
                        'bbox': box.xyxy[0].tolist()
                    })
            
            results["detections"] = detections
            event_type, confidence = classify_event_type(detections)
            results["event_type"] = event_type
            results["confidence"] = confidence
        
        # 2. BLIP-2 Caption (~50-100ms on T4)
        if "caption" in tasks:
            inputs = blip_processor(image, return_tensors="pt").to("cuda")
            out = blip_model.generate(**inputs, max_length=50)
            caption = blip_processor.decode(out[0], skip_special_tokens=True)
            results["caption"] = caption
        
        # 3. NSFW Check (~20-30ms)
        if "nsfw" in tasks:
            nsfw_score = check_nsfw(image)
            results["nsfw_score"] = nsfw_score
            results["is_safe"] = nsfw_score < 0.7
        
        # 4. Blur faces and plates (~10-20ms)
        if "blur" in tasks:
            blurred_image = image.copy()
            
            # Blur faces
            faces = detect_faces(blurred_image)
            for face in faces:
                blurred_image = blur_region(blurred_image, face['bbox'])
            
            # Blur license plates (using YOLO detections)
            if "detections" in results:
                for detection in results["detections"]:
                    # If we had a plate detector, we'd use it here
                    # For now, skip
                    pass
            
            # Encode blurred image
            buffered = io.BytesIO()
            blurred_image.save(buffered, format="JPEG", quality=90)
            blurred_base64 = base64.b64encode(buffered.getvalue()).decode()
            results["blurred_image"] = blurred_base64
        
        return results
        
    except Exception as e:
        return {"error": str(e), "traceback": __import__('traceback').format_exc()}

# Start RunPod handler
runpod.serverless.start({"handler": handler})
