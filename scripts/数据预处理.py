import os
import requests
import zipfile
import shutil
import numpy as np
from pathlib import Path
from PIL import Image
import xml.etree.ElementTree as ET

DATASET_URL = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"
#项目根目录,Path(__file__)表示获取当前脚本文件的路径，parent.parent表示获取当前脚本文件的上一级目录的上一级目录，也就是整个项目文件的路径
project_root = Path(__file__).parent.parent
#YOLO格式数据集目录,当数据定义转换为YOLO格式时，会将数据保存到这个目录下
yolo_data_root = project_root / "data" / "PennFudanPed_YOLO"
#类别映射，数据集中只有一个类别person，对应ID是0
CLASS_MAPPING = {"person": 0}
#训练和验证集划分比例，9:1
ratio = 0.9
#划分比例的随机种子，每次划分时，划分结果都一样
SEED = 42

#转格式
#定义函数mask_to_yolo_polygon，将掩码转换为YOLO分割所需的多边形点格式
def mask_to_yolo_polygon(mask_path, img_width, img_height):   
    #函数接受三个参数，mask_path是掩码文件路径，img_width和img_height是图片的宽度和高度
    mask = Image.open(mask_path)    #利用图像处理库PIL中的Image.open函数打开掩码文件，将其转换为图像对象
    mask_np = np.array(mask)
    
    #提取行人区域掩码非零像素坐标
    #np.where()函数返回掩码数组中所有非零元素(mask_np > 0)的索引，即行人区域的像素坐标,返回的是一个元组，元组中每个元素都是一个数组，对应掩码数组的一个维度的索引
    #np.column_stack()函数将这些索引数组按列堆叠起来，形成一个二维数组，每一行都是一个像素的坐标
    coords = np.column_stack(np.where(mask_np > 0))
    #如果掩码中没有非零像素，即没有行人区域，返回空列表
    if len(coords) == 0:
        return []
    
    y_coords, x_coords = coords[:, 0], coords[:, 1]
    #最多保留50个点
    step = max(1, len(x_coords) // 50)  
    x_coords = x_coords[::step] / img_width
    y_coords = y_coords[::step] / img_height
    
    #拼接为YOLO格式的点字符串
    contour_points = [f"{x:.6f} {y:.6f}" for x, y in zip(x_coords, y_coords)]
    return contour_points

def main():
    #检查下载的数据集
    zip_path = project_root / "PennFudanPed.zip"
    raw_data_dir = project_root / "PennFudanPed"
    
    if raw_data_dir.exists():
        print("已下载数据集，跳过下载")
    else:
        print("没有下载数据集，开始自动下载保存。")
        exit(1)   #终止程序运行并返回状态码
    
    #划分训练集和验证集
    np.random.seed(SEED)
    image_files = list((raw_data_dir / "PNGImages").glob("*.png"))
    np.random.shuffle(image_files)
    train_num = int(ratio * len(image_files))
    train_files = image_files[:train_num]
    val_files = image_files[train_num:]
    print(f"数据集划分完成，训练集{len(train_files)}张，验证集{len(val_files)}张")

    #创建所有需要的文件目录
    (yolo_data_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (yolo_data_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (yolo_data_root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (yolo_data_root / "labels" / "val").mkdir(parents=True, exist_ok=True)

    #转换标注格式并复制图片
    for split, files in [("train", train_files), ("val", val_files)]:
        img_dst_dir = yolo_data_root / "images" / split
        label_dst_dir = yolo_data_root / "labels" / split
        
        for img_path in files:
            #复制图片到YOLO目录
            shutil.copy(img_path, img_dst_dir / img_path.name)
            
            #用掩码处理标注
            img_name = img_path.stem
            img = Image.open(img_path)
            img_w, img_h = img.size
            
            #检查掩码文件是否存在
            mask_path = raw_data_dir / "PedMasks" / f"{img_name}_mask.png"
            if not mask_path.exists():
                print(f"跳过{img_name}，掩码文件不存在")
                os.remove(img_dst_dir / img_path.name)
                continue
            
            #从掩码生成YOLO分割标注
            contour_points = mask_to_yolo_polygon(mask_path, img_w, img_h)
            if contour_points:
                #YOLO分割标注格式，class_id x1 y1 x2 y2 ...
                annotation = f"{0} {' '.join(contour_points)}"  # 0是行人类别ID
                #保存标注文件
                with open(label_dst_dir / f"{img_name}.txt", "w") as f:
                    f.write(annotation)
            else:
                print(f"跳过{img_name}，掩码无有效轮廓")
                os.remove(img_dst_dir / img_path.name)
    
    print("数据集转换完成。\nYOLO格式数据保存到:", yolo_data_root)

if __name__ == "__main__":
    main()