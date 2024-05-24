import os
import shutil

# 指定要分類的檔案(jpg)
search_folder = 'C:/Users/sheng/Desktop/專題_資料/Food 100/UECFOOD100/55'
# 指定新的路徑(jpg)
target_folder = 'C:/Users/sheng/Desktop/測試dataests/55_fried chicken/images'

# 指定要分類的檔案(txt)
annotation_file = 'C:/Users/sheng/Desktop/專題_資料/Food 100/UECFOOD100/55/bb_info.txt'
# 指定新的路徑(txt)
output_folder = 'C:/Users/sheng/Desktop/測試dataests/55_fried chicken/labels'

# 指定要更改的 class_id
new_class_id = '16'

def split_and_modify_annotations(annotation_file):
    with open(annotation_file, 'r') as file:
        lines = file.readlines()

    for line in lines:
        label = line.strip().split()
        class_id = new_class_id
        content = ' '.join(label[1:])

        # 建立新的标注文件
        output_file = os.path.join(output_folder, f'{label[0]}.txt')
        with open(output_file, 'w') as file:
            file.write(f'{class_id} {content}')


# 建立文件夹
os.makedirs(target_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

# 查找 jpg 文件并移动到目标文件夹
for root, dirs, files in os.walk(search_folder):
    for file in files:
        if file.endswith('.jpg'):
            src_file = os.path.join(root, file)
            shutil.copy(src_file, target_folder)

# 分隔标注文件并生成新的标注文件
split_and_modify_annotations(annotation_file)
