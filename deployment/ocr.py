import os
import tempfile
from io import BytesIO

import numpy as np
import requests
import torch
from crnn import CRNN
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

# ---------------------------- CONFIG ----------------------------
app = FastAPI(title="OCR Detection API (FastAPI Version)")

TEXT_DET_MODEL_PATH = r"C:/Users/ASUS/Downloads/TGMT_CK/TGMT_CK/runs/detect/train3/weights/best.pt"
OCR_MODEL_PATH = r"C:/Users/ASUS/Downloads/TGMT_CK/TGMT_CK/models/ocr_crnn.pt"

CHARS = "0123456789abcdefghijklmnopqrstuvwxyz-"
CHAR_TO_IDX = {char: idx + 1 for idx, char in enumerate(sorted(CHARS))}
IDX_TO_CHAR = {idx: char for char, idx in CHAR_TO_IDX.items()}

HIDDEN_SIZE = 256
N_LAYERS = 3
DROPOUT_PROB = 0.2
UNFREEZE_LAYERS = 3

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------- MODELS ----------------------------
print("🔹 Loading YOLO text detection model...")
det_model = YOLO(TEXT_DET_MODEL_PATH)

print("🔹 Loading OCR CRNN recognition model...")
reg_model = CRNN(
    vocab_size=len(CHARS),
    hidden_size=HIDDEN_SIZE,
    n_layers=N_LAYERS,
    dropout=DROPOUT_PROB,
    unfreeze_layers=UNFREEZE_LAYERS,
)
reg_model.load_state_dict(torch.load(OCR_MODEL_PATH, map_location=device))
reg_model.to(device)
reg_model.eval()

# ---------------------------- TRANSFORM ----------------------------
transform = transforms.Compose(
    [
        transforms.Resize((100, 420)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

# ---------------------------- FUNCTIONS ----------------------------

def decode(encoded_sequences, idx_to_char, blank_char="-"):
    decoded_sequences = []
    for seq in encoded_sequences:
        decoded_label = []
        prev_char = None
        for token in seq:
            if token != 0:
                char = idx_to_char[token.item()]
                if char != blank_char:
                    if char != prev_char or prev_char == blank_char:
                        decoded_label.append(char)
                prev_char = char
        decoded_sequences.append("".join(decoded_label))
    return decoded_sequences


def text_recognition(img: Image.Image):
    """Recognize text from a cropped image"""
    transformed_image = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = reg_model(transformed_image).cpu()
    text = decode(logits.permute(1, 0, 2).argmax(2), IDX_TO_CHAR)
    return text


def process_image(image_data: bytes):
    """Perform OCR pipeline: detection + recognition"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            temp_file.write(image_data)
            temp_file_path = temp_file.name

        results = det_model(temp_file_path, verbose=False)[0]
        bboxes, classes, names, confs = (
            results.boxes.xyxy.tolist(),
            results.boxes.cls.tolist(),
            results.names,
            results.boxes.conf.tolist(),
        )

        image = Image.open(temp_file_path)
        predictions = []

        for bbox, cls_idx, conf in zip(bboxes, classes, confs):
            x1, y1, x2, y2 = bbox
            name = names[int(cls_idx)]
            cropped_image = image.crop((x1, y1, x2, y2))
            text = text_recognition(cropped_image)
            predictions.append((bbox, name, conf, text[0]))

        os.unlink(temp_file_path)
        return predictions, image

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")


def draw_predictions(image, predictions):
    """Draw boxes and labels on image"""
    image_array = np.array(image)
    annotator = Annotator(image_array, font="Arial.ttf", pil=False)

    predictions = sorted(predictions, key=lambda x: x[0][1])  # sort by Y coordinate

    for bbox, class_name, conf, text in predictions:
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        color = colors(hash(class_name) % 20, True)
        label = f"{class_name[:3]}{conf:.1f}:{text}"
        annotator.box_label([x1, y1, x2, y2], label, color=color, txt_color=(255, 255, 255))

    return Image.fromarray(annotator.result())

# ---------------------------- API ROUTES ----------------------------

@app.get("/ocr")
async def ocr_from_url(image_url: str):
    """Perform OCR from an image URL"""
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        image_data = response.content
        predictions, image = process_image(image_data)
        annotated = draw_predictions(image, predictions)

        stream = BytesIO()
        annotated.save(stream, format="PNG")
        stream.seek(0)

        return Response(content=stream.getvalue(), media_type="image/png")
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Error downloading image: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ocr/upload")
async def ocr_from_upload(file: UploadFile = File(...)):
    """Perform OCR from an uploaded image"""
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        content = await file.read()
        predictions, image = process_image(content)
        annotated = draw_predictions(image, predictions)

        stream = BytesIO()
        annotated.save(stream, format="PNG")
        stream.seek(0)

        return Response(content=stream.getvalue(), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing uploaded file: {e}")
