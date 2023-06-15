from classes import classes as my_classes
import os
import shutil
import json
import re

source_dir = ".\\labeled_resized\\"
target_dir = ".\\picked_classes\\"
desc_dir = ".\\picked_classes_desc\\"

# for c in my_classes:
#     if os.path.exists(source_dir + c):
#         print(source_dir + c)
#         shutil.copytree(source_dir + c, target_dir + c)


path = u'.\\한국형 사물 이미지\\유적건조물\\'

def make_subset():
    dir_dict = {}
    for f in os.listdir(path):    
        f = f.split('.')
        if f[0] not in dir_dict:
            dir_dict[f[0]] = 1
        else:
            dir_dict[f[0]]+=1

    classes = {}
    i = 0
    images = {}
    for key, value in dir_dict.items():
        if value == 2:
            num_str =  '\\' + str(int(key.split('_')[0]))
            for f in os.listdir(path + key + num_str):
                his_class = f.split('_')[2]
                if his_class not in classes:
                    classes[his_class] = set()
                classes[his_class].add(f.split('_')[-2])

                image_dir = ""
                cur_class = ""
                for c in my_classes:
                    if c.split("_")[-1] == f.split("_")[-1]:
                        image_dir = path + key + num_str + os.sep + f
                        cur_class = c
                        # print(image_dir)
                        # print(c)
                        break

                if image_dir == "": continue

                for file in os.listdir(image_dir):
                    if 'json' in file:
                        file = path + key + num_str + os.sep + f + os.sep + file
                        with open(file, encoding='UTF8') as ff:
                            j = json.load(ff)
                            description = j["regions"][0]['sem_ext'][0]["value"]
                            special_char = '「」≪≫;《》〉〈\\/*?"<>隆不靈六螺蓮·[]|-․()'
                            for c in special_char:
                                description = description.replace(c,'')
                            description = re.compile("[一-龥]").sub('', description)
                            target_file = open(desc_dir + cur_class+".txt", 'w', encoding="UTF8")
                            target_file.write(description)

                        break

# prepare other language description
def make_other_lan():
    import googletrans
    tranlator = googletrans.Translator()

    trans = True
    for f in os.listdir(desc_dir):
        with open(desc_dir+f, encoding="UTF8") as source:
            text = source.read()
            print(text)
            
            ja = tranlator.translate(text, dest='ja', src='ko')
            en = tranlator.translate(text, dest='en', src='ko')
            cn = tranlator.translate(text, dest='zh-cn', src='ko')

            lan = {"en":en.text, "ja":ja.text, "cn":cn.text}
            for key, value in lan.items():
                with open(desc_dir+f.split(".")[0]+"_"+key+".txt", encoding="UTF8", mode="w") as file:
                    file.write(value)


make_other_lan()