import os
import random
import shutil

def create_folders(output_folder):
    # 創建資料夾A到E
    for folder_name in ['A', 'B', 'C', 'D', 'E']:
        folder_path = os.path.join(output_folder, folder_name)
        os.makedirs(folder_path, exist_ok=True)

def main(input_folder, output_folder):
    # 取得所有.jpg檔案
    image_files = [f for f in os.listdir(input_folder) if f.endswith(".jpg")]

    # 隨機排序檔案
    random.shuffle(image_files)

    # 確認檔案數量大於0
    if len(image_files) == 0:
        print("Error: 沒有找到.jpg檔案")
        return

    # 計算每個資料夾應有的檔案數量
    files_per_folder = len(image_files) // 5

    # 分組
    for i in range(5):
        start_index = i * files_per_folder
        end_index = (i + 1) * files_per_folder if i < 4 else None
        group_images = image_files[start_index:end_index]

        # 新增資料夾A到E
        folder_name = chr(ord('A') + i)
        destination_folder = os.path.join(output_folder, folder_name)

        # 移動檔案到目標資料夾
        for image_file in group_images:
            text_file = os.path.splitext(image_file)[0] + ".txt"

            image_source = os.path.join(input_folder, image_file)
            text_source = os.path.join(input_folder, text_file)

            image_destination = os.path.join(destination_folder, image_file)
            text_destination = os.path.join(destination_folder, text_file)

            os.makedirs(destination_folder, exist_ok=True)
            shutil.move(image_source, image_destination)
            shutil.move(text_source, text_destination)

if __name__ == "__main__":
    input_folder = "/home/dd/yolov7/yolov7/data/dataests/in"
    output_folder = "/home/dd/yolov7/yolov7/data/dataests/5Ford"  # 替換成實際的輸出資料夾路徑

    create_folders(output_folder)
    main(input_folder, output_folder)