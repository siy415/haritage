# -*- Encoding: UTF-8 -*- #
import os, sys
import json
from PIL import Image
import  shutil
from setting import *

path = u'.\\한국형 사물 이미지\\유적건조물\\'

dir_dict = {}
for f in os.listdir(path):
    # print(path+f)
    
    f = f.split('.')
    if f[0] not in dir_dict:
        dir_dict[f[0]] = 1
    else:
        dir_dict[f[0]]+=1

classes = {}
i = 0
images = {}

# rename zip file to reduce the capacity of the compressed files.
if DELETE_ZIP:
    for key, value in dir_dict.items():
        if value == 2:
            # print(path + key + '.zip')
            # os.remove(path + key + '.zip')
            # f = open(path + key + '.zip', 'w')
            # f.write("1")
            # f.close()
            num_str =  '\\' + str(int(key.split('_')[0]))
            for f in os.listdir(path + key + num_str):
                his_class = f.split('_')[2]
                if his_class not in classes:
                    classes[his_class] = set()
                classes[his_class].add(f.split('_')[-2])

                image_dir = path + key + num_str + os.sep + f

                for file in os.listdir(image_dir):
                    if file.split('.')[0] not in images:
                        images[file.split('.')[0]] = {}
                    if 'json' in file:
                        images[file.split('.')[0]]['json'] = path + key + num_str + os.sep + f + os.sep + file
                    else:
                        images[file.split('.')[0]]['jpg'] = path + key + num_str + os.sep + f + os.sep + file


if SUBSET_FLAG:
    target_dir = '.\\labeled2\\'
    cur_dir = ''
    for key, value in images.items():
        if 'jpg' not in value or 'json' not in value: continue
        out_dir = target_dir + value['jpg'].split('\\')[5].split('_')[2] + "_" + value['jpg'].split('\\')[5].split('_')[-2] + '_' + value['jpg'].split('\\')[5].split('_')[-1] + os.sep
        if (cur_dir != out_dir):
            cur_dir=out_dir
            print("[crop]:", cur_dir)

        if not os.path.exists(out_dir): os.mkdir(out_dir)

        if not os.path.exists(out_dir + value['jpg'].split('\\')[-1]): 
            with open(value['json'], encoding='UTF8') as f:
                j = json.load(f)
                image = Image.open(value['jpg'])
                try:
                    boxcorners = j["regions"][0]['boxcorners']
                    crop_image = image.crop((boxcorners[0],boxcorners[1],boxcorners[2]-1,boxcorners[3]-1))
                    # print(out_dir + value['jpg'].split('\\')[-1])
                    crop_image.save(out_dir + value['jpg'].split('\\')[-1])
                except ValueError as e:
                    print(e, ":", boxcorners)
                except OSError as e:
                    print(e, ":", value['jpg'], boxcorners)

if CROP_IMAGE:
    cur_dir = ''
    croped_dir = '.\\labeled_resized\\'
    for key, value in images.items():
        if 'jpg' not in value or 'json' not in value: continue
        
        out_dir = target_dir + value['jpg'].split('\\')[5].split('_')[2] + "_" + value['jpg'].split('\\')[5].split('_')[-2] + '_' + value['jpg'].split('\\')[5].split('_')[-1] + os.sep
        resize_out_dir = croped_dir + value['jpg'].split('\\')[5].split('_')[2] + "_" + value['jpg'].split('\\')[5].split('_')[-2] + '_' + value['jpg'].split('\\')[5].split('_')[-1] + os.sep
        num = 1
        if (cur_dir != out_dir):
            print(cur_dir)
            cur_dir=out_dir
            try:
                images_dir = os.listdir(cur_dir)
                # delete directory which has few images
                if len(images_dir) < 500: 
                    print("[delete] {}, len: {}".format(cur_dir, len(images_dir)))
                    shutil.rmtree(cur_dir)

                elif len(images_dir) == 0 : continue
                else:
                    if not os.path.exists(resize_out_dir): os.mkdir(resize_out_dir)
                    for f in images_dir:
                        if not os.path.exists(resize_out_dir + f): 
                            cur_img = Image.open(out_dir + f)
                            size = cur_img.size
                            max_val = max(size)
                            min_idx = 0 if size[0] == min(size) else 1
                            resize_ratio = 256.0 / max_val
                            resize_value = (int(size[0] * resize_ratio), int(size[1] * resize_ratio))
                            resized_img = cur_img.resize(resize_value)
                            resized_img_size = resized_img.size
                            
                            padding_size = 256 - resized_img_size[min_idx]
                            padding_size_down = round(padding_size / 2)
                            padding_size_up = padding_size_down if padding_size_down & 0x1 == 0 else padding_size_down + 1

                            result = Image.new(resized_img.mode, (256,256), (0,0,0))

                            start_cord = (padding_size_down, 0) if min_idx == 0 else  (0, padding_size_down)
                            # result.paste(resized_img, (256-resized_img_size[0]//2, 256-resized_img_size[1]//2))
                            result.paste(resized_img, start_cord)

                            
                            result.save(resize_out_dir + f)

                        if SHOW_IMAGE:
                            print("show Image")
                            resized_img.show()
                            result.show()
                            show = False


            except FileNotFoundError as e:
                pass
                # print(e, "Already deleted: {}".format(cur_dir))


        num_del_interval = len(images_dir) / 1000

        


        for f in images_dir:
            if(num < num_del_interval and num_del_interval > 1):
                os.remove(out_dir + f)
            num = (1 if(num == num_del_interval) else num + 1)