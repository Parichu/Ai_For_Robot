# test_model_standalone.py

import cv2
from ultralytics import YOLO
import time



print("กำลังโหลดโมเดล AI...")
# --- 1. โหลดโมเดล ---
model_path = "/home/parichu/AiForRobot/Ai_For_Robot/AI_Model/best.pt" 
model = YOLO(model_path)
print(f"โหลดโมเดล {model_path} สำเร็จ")

# --- 2. โหลดรูปภาพสำหรับทดสอบ ---
image_path = "/home/parichu/Ai/all_workspace/ai_module2/images/received_image_2.jpeg"
img = cv2.imread(image_path)

if img is None:
    print(f"ไม่สามารถอ่านไฟล์รูปภาพได้ที่: {image_path}")
else:
    print(f"กำลังทดสอบ detect รูปภาพ: {image_path}")
    
    # --- 3. สั่งให้โมเดล Detect ---
    # model(img) คือคำสั่งหลักในการรัน detection
    start_time = time.time()
    results = model(img)
    end_time = time.time()
    
    print(f"Detection เสร็จสิ้นใน {end_time - start_time:.4f} วินาที")

    # --- 4. แสดงผลลัพธ์ ---
    # 4.1 พิมพ์ผลลัพธ์ (เช่น class, confidence, พิกัด) ลงบน Terminal
    print("\n--- ผลลัพธ์ (Boxes) ---")
    print(results[0].boxes)
    print("------------------------\n")

    # 4.2 วาดกรอบลงบนรูปภาพ
    # .plot() เป็นฟังก์ชันของ YOLO ที่ช่วยวาดผลลัพธ์ลงบนภาพให้เราอัตโนมัติ
    annotated_frame = results[0].plot()

    # 4.3 แสดงรูปภาพที่มีกรอบ Bounding Box
    print("กำลังแสดงผลลัพธ์... (กดปุ่ม 'q' บนหน้าต่างรูปภาพเพื่อปิด)")
    cv2.imshow("AI Detection Test (Standalone)", annotated_frame)
    
    # รอจนกว่าจะมีการกดปุ่ม
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("ปิดโปรแกรมทดสอบ")
