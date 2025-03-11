import torch
from ultralytics import YOLO
def main():
    # Xác định thiết bị: GPU nếu có, ngược lại CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Kiểm tra có sử dụng gpu không
    print("Using device:", device)
    print(torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    torch.cuda.empty_cache()
    #Tạo model YOLO, phiên bản model có thể tự động load
    model = YOLO("yolo11n.pt")

    data_path = 'dataset2/data.yaml'
    try:
        results = model.train(data=data_path, epochs=10, imgsz=640, device=device, batch = 8)
        print("Training results:", results)
    except Exception as e:
        print("Error:", e)


if __name__ == '__main__':
    main()
