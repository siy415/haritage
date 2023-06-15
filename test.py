from make_model import *
from setting import *
from PIL  import Image


def preprocessing(path):
    cur_img = Image.open(path)
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

    # result.show()

    return result


def evaluate_model(model, imge_file_path):
    image = preprocessing(imge_file_path)
    toTensor = torchvision.transforms.ToTensor()
    model.eval()
    image_tensor = toTensor(image)
    image_tensor = torch.unsqueeze(image_tensor, 0)
    
    pred = model(image_tensor.to(device))

    whole_data = get_dataset_from_dir(data_dir, transform)
    pred_class = whole_data.classes[torch.max(pred, 1)[1][0]]
    desc_file = open(desc_dir + pred_class+".txt", 'r', encoding="UTF8")
    desc = desc_file.readline()
    print("Heritage: {}".format(pred_class.split('_')[-1]))
    print("description: {}".format(desc))

    result = {"Heritage" : pred_class.split('_')[-1],
              "description" : desc}

    return result

data_dir = '.\\picked_classes'
desc_dir = 'F:\\image\\picked_classes_desc\\'

transform = transforms.Compose([
                                        transforms.ToTensor(),
                                    ])



my_model = torch.load('.\\model.pth')

path = "F:\\image\\한국형 사물 이미지\\유적건조물\\64_유적건조물\\64\\HF010785_유적건조물_교통통신_교통통신시설_우체국_인천우체국\\HF010785_0002_0002.jpg"

print(evaluate_model(my_model, path))
