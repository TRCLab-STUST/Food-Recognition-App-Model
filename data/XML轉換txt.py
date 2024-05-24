import os
import xml.etree.ElementTree as ET

def convert_xml_to_txt(xml_folder, output_folder, class_id=0):
    # 確保輸出資料夾存在
    os.makedirs(output_folder, exist_ok=True)

    # 迭代處理資料夾中的每個XML檔案
    for xml_file in os.listdir(xml_folder):
        if xml_file.endswith('.xml'):
            xml_path = os.path.join(xml_folder, xml_file)
            txt_file = os.path.splitext(xml_file)[0] + '.txt'
            txt_path = os.path.join(output_folder, txt_file)

            # 開啟XML檔案
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # 開啟txt檔案並寫入資料
            with open(txt_path, 'w') as f:
                for obj in root.findall('object'):
                    class_name = obj.find('name').text
                    bbox = obj.find('bndbox')
                    xmin = bbox.find('xmin').text
                    ymin = bbox.find('ymin').text
                    xmax = bbox.find('xmax').text
                    ymax = bbox.find('ymax').text

                    # 將類別名稱和座標寫入txt檔案
                    line = f'{class_id} {xmin} {ymin} {xmax} {ymax}\n'
                    f.write(line)

# 指定包含XML標註檔案的資料夾路徑
xml_folder = 'C:/Users/sheng/Desktop/測試dataests/0_XML檔案'
# 指定轉換後的txt檔案要儲存的資料夾路徑
output_folder = 'C:/Users/sheng/Desktop/測試dataests/0_XML檔案/XML_DATA'

# 指定第一個類別的class_id
class_id = 2

# 呼叫函式，執行轉換
convert_xml_to_txt(xml_folder, output_folder, class_id)
