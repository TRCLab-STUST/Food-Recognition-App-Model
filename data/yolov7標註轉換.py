import os

def convert_labels_to_yolov7_format(folder):
    for filename in os.listdir(folder):
        if filename.endswith('.txt'):
            filepath = os.path.join(folder, filename)
            with open(filepath, 'r') as file:
                lines = file.readlines()

            with open(filepath, 'w') as file:
                for line in lines:
                    try:
                        label = line.strip().split()
                        class_id = int(label[0])  # 假設類別 ID 在第一列
                        x = float(label[1])
                        y = float(label[2])
                        x2 = float(label[3])
                        y2 = float(label[4])

                        # 計算 YOLOv7 格式的邊界框座標
                        img_width = 800  # 替換為您的圖片寬度
                        img_height = 800  # 替換為您的圖片高度
                        width = x2 - x
                        height = y2 - y
                        x_center = (x + width / 2) / img_width
                        y_center = (y + height / 2) / img_height
                        w_normalized = width / img_width
                        h_normalized = height / img_height

                        # 將標註以 YOLOv7 格式寫入文件
                        file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_normalized:.6f} {h_normalized:.6f}\n")
                    except ValueError:
                        print(f"Error processing line: {line}")

# 將資料夾中的標註轉換為 YOLOv7 格式
train_folder = 'C:/Users/sheng/Desktop/測試dataests/0_XML檔案/XML_DATA'
#train_folder = 'C:/Users/sheng/Desktop/測試dataests/4_chicken-egg on rice/labels'
convert_labels_to_yolov7_format(train_folder)
