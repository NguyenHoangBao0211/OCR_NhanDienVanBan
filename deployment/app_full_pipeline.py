# Gom 2 bước trên (YOLO + CRNN) thành API OCR hoàn chỉnh (gom crnn + object_detection)  
import os
import tempfile
from io import BytesIO
import numpy as np
import torch
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from crnn import CRNN  # Đảm bảo file crnn.py có class CRNN của bạn
import warnings

warnings.filterwarnings("ignore")

# Khởi tạo ứng dụng FastAPI
app = FastAPI(title="OCR Full Pipeline API")

# =====================
# 🔧 1️⃣ CẤU HÌNH (CONFIG)
# =====================
TEXT_DET_MODEL_PATH = r"C:/Users/ASUS/Downloads/TGMT_CK/TGMT_CK/runs/detect/train3/weights/best.pt"
OCR_MODEL_PATH = r"C:/Users/ASUS/Downloads/TGMT_CK/TGMT_CK/models/ocr_crnn.pt"

CHARS = "0123456789abcdefghijklmnopqrstuvwxyz-"
IDX_TO_CHAR = {i + 1: c for i, c in enumerate(sorted(CHARS))}  # anh xạ index -> ký tự
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"        # chọn thiết bị GPU nếu có

# =====================
# 🧠 2️⃣ TẢI MÔ HÌNH (LOAD MODELS)
# =====================
print("🔹 Loading YOLO text detection model...")
det_model = YOLO(TEXT_DET_MODEL_PATH)    # tải mô hình YOLO đã train để phát hiện text

print("🔹 Loading OCR CRNN recognition model...")
HIDDEN_SIZE = 256
N_LAYERS = 3
DROPOUT = 0.2
UNFREEZE_LAYERS = 3

# Tạo mô hình CRNN (phải khớp cấu hình khi train)
ocr_model = CRNN(
    vocab_size=len(CHARS),
    hidden_size=HIDDEN_SIZE,
    n_layers=N_LAYERS,
    dropout=DROPOUT,
    unfreeze_layers=UNFREEZE_LAYERS
).to(DEVICE)

# nạp trọng số đã train của model CRNN
ocr_model.load_state_dict(torch.load(OCR_MODEL_PATH, map_location=DEVICE))
ocr_model.eval()   # chuyển sang chế độ inference (tắt dropout, batchrom..)


# =====================
# 🔤 3️⃣ HÀM TIỆN ÍCH (UTILITY FUNCTIONS)
# =====================
def decode(encoded_sequences, idx_to_char, blank_char="-"):
    """
    Giải mã đầu ra của mô hình CRNN (CTC decoding thô)
    - Loại bỏ ký tự trống (blank)
    - Gộp ký tự trùng lặp liên tiếp
    """

    decoded_sequences = []
    for seq in encoded_sequences:
        decoded_label = []
        prev_char = None
        for token in seq:
            if token != 0:    # bỏ padding
                char = idx_to_char.get(token.item(), "")
                if char != blank_char:
                    if char != prev_char or prev_char == blank_char:
                        decoded_label.append(char)
                prev_char = char
        decoded_sequences.append("".join(decoded_label))
    return decoded_sequences


def recognize_text(cropped_image):
    """
   Hàm nhận dạng chữ từ một ảnh đã crop (word image)
    1️⃣ Tiền xử lý ảnh (resize, grayscale, normalize)
    2️⃣ Dự đoán bằng CRNN
    3️⃣ Giải mã đầu ra thành chuỗi ký tự
    """

    transform = transforms.Compose([
        transforms.Resize((100, 420)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image_tensor = transform(cropped_image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = ocr_model(image_tensor).cpu()
    text = decode(logits.permute(1, 0, 2).argmax(2), IDX_TO_CHAR)  # chọn ký tự xác suất cao nhất
    return text[0]     # trả về chuỗi text duy nhất 


# =====================
# 📸 4️⃣ ĐỊNH NGHĨA API ENDPOINT
# =====================
@app.post("/ocr/upload", response_class=Response)
async def ocr_upload(file: UploadFile = File(...)):
    """API nhận file ảnh → phát hiện text bằng YOLO → nhận dạng chữ bằng CRNN → trả về ảnh đã annotate
    """
    try:
        # Kiểm tra loại file
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Đọc dữ liệu ảnh
        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")

        # Phát hiện vùng chữ bằng YOLO
        results = det_model(image, verbose=False)[0]
        boxes, classes, names, confs = (
            results.boxes.xyxy.tolist(),   # tọa độ bbox [x1, y1, x2, y2]
            results.boxes.cls.tolist(),    # class id
            results.names,                 # tên lớp 
            results.boxes.conf.tolist(),   # độ tin cậy
        )

        # Khởi tạo Annotator để vẽ kết quả
        img_np = np.array(image)
        annotator = Annotator(img_np, font="Arial.ttf", pil=False)

        # Lặp qua từng vùng text phát hiện được
        for box, cls, conf in zip(boxes, classes, confs):
            x1, y1, x2, y2 = [int(v) for v in box]
            cropped = image.crop((x1, y1, x2, y2))        # cắt vùng chữ 
            text = recognize_text(cropped)                # nhận dạng nội dung chữ 
            label = f"{names[int(cls)]} {conf:.2f}: {text}"
            annotator.box_label([x1, y1, x2, y2], label, color=colors(int(cls), True))   # vẽ bbox + label

        # Trả kết quả là ảnh annotated
        annotated_image = Image.fromarray(annotator.result())   # chuyển pilow -> numpy
        buf = BytesIO()
        annotated_image.save(buf, format="PNG")   # lưu ảnh 
        buf.seek(0)

        return Response(content=buf.getvalue(), media_type="image/png")

    except Exception as e:
        # Bắt lỗi chung: file lỗi, model lỗi, định dạng không hợp lệ...
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")


# =====================
# 🚀 Run Command:
# =====================
# python -m uvicorn app_full_pipeline:app --reload --port 8000
