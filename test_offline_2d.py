import cv2
from ultralytics import YOLO
import os
import numpy as np # <-- MỚI: Cần cho việc xử lý mảng

MODEL_PATH = './best5.pt' 

IMAGE_PATH = './submit/rgb/0013.png' 


# 1. Tải mô hình
if not os.path.exists(MODEL_PATH):
    print(f"LỖI: Không tìm thấy model tại: {MODEL_PATH}")
    exit()
    
print(f"Đang tải model từ: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

# 2. Tải ảnh
if not os.path.exists(IMAGE_PATH):  
    print(f"LỖI: Không tìm thấy ảnh tại: {IMAGE_PATH}")
    exit()

print(f"Đang đọc ảnh từ: {IMAGE_PATH}")
img = cv2.imread(IMAGE_PATH) # <-- Ảnh gốc sẽ được vẽ trực tiếp lên

# 3. Chạy dự đoán
print("Đang chạy dự đoán...")
results = model.predict(source=img, conf=0.5) 

# --- ĐẾM, IN SỐ LƯỢNG VÀ IN TỌA ĐỘ (Giữ nguyên) ---

result = results[0] 
object_count = len(result.boxes)
print(f"*** ĐÃ TÌM THẤY: {object_count} ĐỐI TƯỢNG TÁCH BIỆT ***")

if result.masks:
    print("\n--- Tọa độ đường bao (Masks) của các đối tượng ---")
    list_of_masks_xy = result.masks.xy
    
    for i, mask_points_xy in enumerate(list_of_masks_xy):
        print(f"\n[ĐỐI TƯỢNG {i+1}] (Tổng cộng {len(mask_points_xy)} điểm):")
        print(mask_points_xy.astype(int)) 
        
else:
    print("\nLƯU Ý: Model này không trả về 'masks' (đường bao).")
    # ... (phần còn lại giữ nguyên)


# ---------------------------------------------------
# --- 4. VẼ KẾT QUẢ THỦ CÔNG BẰNG OPENCV ---
# (Thay thế cho: img_with_results = result.plot(boxes=False))
# ---------------------------------------------------

if result.masks: # Chỉ vẽ nếu có mask
    
    # Lấy tất cả thông tin cần thiết từ 'result'
    list_of_masks_xy = result.masks.xy
    all_boxes_xyxy = result.boxes.xyxy.cpu().numpy().astype(int)
    confidences = result.boxes.conf.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    class_names = result.names # Đây là dict, ví dụ: {0: 'package'}

    print("\nĐang vẽ thủ công lên ảnh...")

    # Lặp qua từng đối tượng tìm được
    for i in range(len(list_of_masks_xy)):
        
        # A. Lấy tọa độ MASK và chuẩn bị cho OpenCV
        mask_points = list_of_masks_xy[i].astype(np.int32)
        # Chuyển đổi sang định dạng [N, 1, 2] mà cv2.polylines cần
        cv2_mask_points = mask_points.reshape((-1, 1, 2))
        
        # B. Lấy thông tin HỘP BAO (để đặt chữ)
        x1, y1, x2, y2 = all_boxes_xyxy[i]
        
        # C. Lấy thông tin NHÃN (LABEL)
        conf = confidences[i]
        cls_id = class_ids[i]
        label = class_names[cls_id]
        
        label_text = f"{label} ({conf*100:.1f}%)"
        
        # === THỰC HIỆN VẼ LÊN ẢNH 'img' ===
        
        # 1. Vẽ đường bao (Mask) - Màu xanh lá
        cv2.polylines(img, [cv2_mask_points], isClosed=True, color=(0, 255, 0), thickness=2)
        
        # 2. Vẽ nhãn (Text)
        # Đặt chữ ngay phía trên góc trái (x1, y1) của hộp bao
        cv2.putText(img, label_text, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

else:
    print("Không có mask để vẽ thủ công.")

# ---------------------------------------------------

# 5. Hiển thị ảnh trong một cửa sổ pop-up
print("\nĐã dự đoán xong. Đang hiển thị kết quả...")

window_title = f"Ket Qua (Ve thu cong): {object_count} doi tuong (Nhan phim bat ky de dong)"

# SỬA: Hiển thị 'img' (ảnh gốc đã bị vẽ đè lên)
# thay vì 'img_with_results'
cv2.imshow(window_title, img) 

# Đợi người dùng nhấn một phím bất kỳ
cv2.waitKey(0) 

# Đóng tất cả cửa sổ
cv2.destroyAllWindows() 
print("Đã đóng cửa sổ.")