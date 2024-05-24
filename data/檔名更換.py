import os


def rename_files(folder_path, start_number=1, padding_length=5, file_extension=".jpg", new_file_prefix="image"):
    # 確保資料夾存在
    os.makedirs(folder_path, exist_ok=True)

    # 獲取資料夾中的所有檔案
    file_list = os.listdir(folder_path)

    # 確定數字序列的起始值
    current_number = start_number

    # 遍歷每個檔案
    for file_name in file_list:
        # 確認檔案的副檔名是否符合指定的副檔名
        if os.path.splitext(file_name)[1] == file_extension:
            # 建立新的檔名
            new_file_name = f"{new_file_prefix}_{str(current_number).zfill(padding_length)}{file_extension}"

            # 舊的檔案路徑
            old_file_path = os.path.join(folder_path, file_name)

            # 新的檔案路徑
            new_file_path = os.path.join(folder_path, new_file_name)

            # 更改檔案名稱
            os.rename(old_file_path, new_file_path)

            print(f"Renamed file: {file_name} to {new_file_name}")

            # 更新數字序列
            current_number += 1


# 使用範例：
folder_path = "C:/Users/sheng/Desktop/NEW資料集/39_bubble tea 珍珠奶茶/train"  # 替換為實際的資料夾路徑
start_number = 696 # 起始數字
padding_length = 6  # 數字序列的填充長度
file_extension = ".txt"  # 要讀取的副檔名
new_file_prefix = "bubble tea"  # 新的檔名前綴

# 呼叫函式進行重新命名
rename_files(folder_path, start_number, padding_length, file_extension, new_file_prefix)
