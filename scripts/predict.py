import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  
plt.rcParams['axes.unicode_minus'] = False    

#模型权重绝对路径
MODEL_PATH = r"E:\messy_files\UD_lab_test\runs\segment\runs\segment\yolo_model1\weights\best.pt"
#自己导入的网图的路径
CUSTOM_IMG_PATH = r"E:\messy_files\UD_lab_test\lasttry2\网图.jpg"  
#输出保存
OUTPUT_PATH = r"E:\messy_files\UD_lab_test\lasttry2\最终输出"

def plot_only_bbox(img, results):
    """仅绘制边界框和标签，不绘制分割掩码"""
    img_copy = img.copy()
    for r in results:
        boxes = r.boxes
        for box in boxes:
            #边界框坐标获取
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            #类别和置信度获取
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = r.names[cls]
            
            #红色边界框绘制
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)
            #带标签背景绘制
            label = f"{cls_name} {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y1 = y1 - label_size[1] if y1 - label_size[1] > 0 else y1 + label_size[1]
            cv2.rectangle(img_copy, (x1, label_y1 - label_size[1]), 
                          (x1 + label_size[0], label_y1), (0, 0, 255), -1)
            cv2.putText(img_copy, label, (x1, label_y1), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img_copy

def main():
    #检查模型文件是否存在
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        print(f"模型文件不存在，{MODEL_PATH}")
        return
    
    custom_img_path = Path(CUSTOM_IMG_PATH)
    if not custom_img_path.exists():
        print(f"导入图片不存在，{CUSTOM_IMG_PATH}")
        return
    
    #加载模型
    model = YOLO(MODEL_PATH)
    #读取自定义图片
    img_original = cv2.imread(str(custom_img_path))
    if img_original is None:
        print(f"无法读取图片，格式错误或者文件损坏：{CUSTOM_IMG_PATH}")
        return
    img_original_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
    
    #执行预测
    results = model(img_original)  
    result = results[0]
    
    #BBox预测图
    img_only_bbox = plot_only_bbox(img_original, results)
    img_only_bbox_rgb = cv2.cvtColor(img_only_bbox, cv2.COLOR_BGR2RGB)
    
    #Mask叠加效果图
    img_mask_overlay = result.plot()
    img_mask_overlay_rgb = cv2.cvtColor(img_mask_overlay, cv2.COLOR_BGR2RGB)
    
    #拼接成一张组合图
    plt.figure(figsize=(24, 8))
    
    #原图
    plt.subplot(1, 3, 1)
    plt.imshow(img_original_rgb)
    plt.axis("off")
    plt.title("原图", fontsize=16)
    
    #BBox预测图
    plt.subplot(1, 3, 2)
    plt.imshow(img_only_bbox_rgb)
    plt.axis("off")
    plt.title("行人边界框识别结果", fontsize=16)
    
    #Mask叠加效果图
    plt.subplot(1, 3, 3)
    plt.imshow(img_mask_overlay_rgb)
    plt.axis("off")
    plt.title("行人分割掩码叠加效果", fontsize=16)
    
    plt.tight_layout(pad=3.0)
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")  
    plt.show()
    
    print(f"\n预测完成。组合结果图已保存至：{OUTPUT_PATH}")
    img_name = custom_img_path.stem
    cv2.imwrite(f"{img_name}_仅BBox.jpg", img_only_bbox)
    cv2.imwrite(f"{img_name}_Mask叠加.jpg", img_mask_overlay)

if __name__ == "__main__":
    main()