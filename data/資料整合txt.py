import os

def create_file_list(root_dir, output_file):
    with open(output_file, 'w') as f:
        for folder in ['train', 'valid']:
            folder_path = os.path.join(root_dir, folder)
            image_files = [file for file in os.listdir(folder_path) if file.endswith('.jpg')]
            for img_file in image_files:
                img_path = os.path.join(folder_path, img_file)
                label_file = os.path.join(folder_path, img_file.replace('.jpg', '.txt'))
                if os.path.exists(label_file):
                    f.write(f"{img_path}\n")

if __name__ == '__main__':
    root_directory = 'D:/PycharmProjects/pythonProject_RR/yolov71/data/dataests'  # 將此處替換為您資料集根目錄的路徑
    train_txt_file = 'train.txt'
    valid_txt_file = 'valid.txt'

    create_file_list(root_directory, train_txt_file)
    create_file_list(root_directory, valid_txt_file)

    print(f"train.txt和valid.txt文件已成功創建。")
