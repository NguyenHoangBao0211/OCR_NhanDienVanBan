# Chạy YOLO để phát hiện vùng chữ (text detection)
import os
import tempfile
from io import BytesIO

import numpy as np
import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

app = FastAPI()


class ObjectDetection:
    def __init__(self):
        # Đường dẫn đến mô hình YOLO của bạn
        self.model = YOLO("D:/Du_Lieu_Hoc/TGMT/TGMT_CK/runs/detect/train3/weights/best.pt")

    def detect(self, image_path: str):
        try:
            # Thực hiện phát hiện đối tượng
            results = self.model(image_path, verbose=False)[0]
            return (
                results.boxes.xyxy.tolist(),
                results.boxes.cls.tolist(),
                results.names,
                results.boxes.conf.tolist(),
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing image: {e}")


# Tạo instance của mô hình
object_detector = ObjectDetection()


async def process_image(image_data: bytes) -> Response:
    """Hàm xử lý chung cho ảnh (dùng cho URL và upload file)"""
    try:
        # Tạo file tạm
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            temp_file.write(image_data)
            temp_file_path = temp_file.name

        # Gọi model phát hiện đối tượng
        bboxes, classes, names, confs = object_detector.detect(temp_file_path)

        # Mở ảnh từ file tạm
        image = Image.open(temp_file_path)
        image_array = np.array(image)

        # Annotator vẽ kết quả
        annotator = Annotator(image_array, font="Arial.ttf", pil=False)

        for box, cls, conf in zip(bboxes, classes, confs):
            c = int(cls)
            label = f"{names[c]} {conf:.2f}"
            annotator.box_label(box, label, color=colors(c, True))

        # Chuyển ảnh đã annotate về dạng bytes
        annotated_image = Image.fromarray(annotator.result())
        file_stream = BytesIO()
        annotated_image.save(file_stream, format="PNG")
        file_stream.seek(0)

        # Xoá file tạm
        os.unlink(temp_file_path)

        return Response(content=file_stream.getvalue(), media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")


@app.get("/detect", response_class=Response)
async def detect_url(image_url: str):
    """Endpoint phát hiện đối tượng từ URL ảnh"""
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        return await process_image(response.content)
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Error downloading image: {e}")


@app.post("/detect/upload", response_class=Response)
async def detect_upload(file: UploadFile = File(...)):
    """Endpoint phát hiện đối tượng từ file ảnh tải lên"""
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        content = await file.read()
        return await process_image(content)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing uploaded file: {e}")


# Chạy bằng lệnh:
# uvicorn ocr:app --reload --port 8000
