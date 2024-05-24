import os

def modify_class_id(folder_path, new_class_id):
    # 獲取資料夾中的所有 txt 檔案
    file_list = [file for file in os.listdir(folder_path) if file.endswith('.txt')]

    # 遍歷每個檔案
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)

        # 開啟檔案進行讀取和修改
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # 遍歷每一行
        for i in range(len(lines)):
            line = lines[i].strip()
            if line:
                # 分割行的內容
                class_id, rest_of_line = line.split(' ', 1)
                # 將 class_id 替換為指定的新數字
                new_line = str(new_class_id) + ' ' + rest_of_line
                lines[i] = new_line + '\n'

        # 將修改後的內容保存回檔案
        with open(file_path, 'w') as file:
            file.writelines(lines)

        print(f"Modified file: {file_name}")

# 使用範例：
folder_path = 'C:/Users/sheng/Desktop/NEW資料集/38_fried sweet potato balls 地瓜球/train'
#folder_path = 'C:/Users/sheng/Desktop/NEW資料集/38_fried sweet potato balls 地瓜球/valid'
new_class_id = 38 # 替換為指定的新 class_id

# 呼叫函數進行處理
modify_class_id(folder_path, new_class_id)
