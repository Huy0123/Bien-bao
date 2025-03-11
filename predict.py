import torch
from ultralytics import YOLO
import cv2


def main():
    # Xác định thiết bị: GPU nếu có, ngược lại CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model đã train (thay 'runs/train/exp/weights/best.pt' bằng đường dẫn đúng)
    model = YOLO('runs/detect/train/weights/best.pt')

    # Chạy cái này nếu muốn test ảnh
    # Chỉnh sửa địa chỉ file ảnh
    #results = model.predict('gt.jpg', device=device, save=True, imgsz=640)
    # Chạy cái này nếu muốn test video
    #results = model.predict('test.mp4', device=device, save=True, imgsz=640)

    # Hiển thị ảnh sau khi nhận diện (chỉ dùng cho test ảnh, có thể xem ảnh trong runs/detect)
    # for result in results:
    #     img = result.plot()
    #     cv2.imshow('Prediction', img)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
