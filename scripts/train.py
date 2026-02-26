from ultralytics import YOLO
from pathlib import Path

#准备配置
#项目根目录
ROOT_DIR = Path(__file__).parent.parent
#数据集配置的文件路径
CONFIG_PATH = ROOT_DIR / "config" / "PennFudan.yaml"
#训练参数
eopchs = 100          
batch_size = 8        
img_size = 640        
device = 0            
model_type = "yolov8s-seg.pt"  

def main():
    #加载预训练好的模型
    model = YOLO(model_type)
    print(f"加载模型完成：{model_type}")
    
    #开始训练
    print("开始训练。")
    results = model.train(
        data=str(CONFIG_PATH),
        epochs=eopchs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        workers=0,
        lr0=0.01,           #初始学习率
        weight_decay=0.001, #权重衰减
        patience=50,         #早停机制
        save=True,           #保存最佳模型
        project="runs/segment", #结果保存目录
        name="yolo_model1",  #实验名称
        exist_ok=True,       #覆盖已有目录
        verbose=True         #打印日志
    )    

    #验证模型
    print("开始验证模型。")
    metrics = model.val()
    print(f"验证集指标：")
    print(f"边界框mAP50: {metrics.box.map50:.4f}")
    print(f"分割mAP50: {metrics.seg.map50:.4f}")

if __name__ == "__main__":
    main()