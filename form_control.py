from make_model import *
from setting import *
from PIL  import Image
import googletrans


class Control():
    def __init__(self):
        self.__image_path = ""
        self.model = torch.load('.\\model.pth')
        self.data_dir = '.\\picked_classes'
        self.desc_dir = 'F:\\image\\picked_classes_desc\\'
        self.transform = transforms.Compose([
                                        transforms.ToTensor(),
                                    ])
        self.desc = ""
        self.title_ko = ""
        self.title_en = ""
        self.title_ja = ""
        self.title_cn = ""

    @property
    def image_path(self):
        return self.__image_path
    
    @image_path.setter
    def image_path(self, path):
        print(path)
        self.__image_path = path
    
    def preprocessing(self, path):
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

        result = Image.new(resized_img.mode, (256,256), (0,0,0))

        start_cord = (padding_size_down, 0) if min_idx == 0 else  (0, padding_size_down)
        # result.paste(resized_img, (256-resized_img_size[0]//2, 256-resized_img_size[1]//2))
        result.paste(resized_img, start_cord)

        # result.show()

        return result

    def evaluate_model(self):
        image = self.preprocessing(self.__image_path)
        toTensor = torchvision.transforms.ToTensor()
        self.model.eval()
        image_tensor = toTensor(image)
        image_tensor = torch.unsqueeze(image_tensor, 0)
        
        pred = self.model(image_tensor.to(device))

        whole_data = get_dataset_from_dir(self.data_dir, self.transform)
        pred_class = whole_data.classes[torch.max(pred, 1)[1][0]]
        desc_file = open(self.desc_dir + pred_class+".txt", 'r', encoding="UTF8")
        self.desc = self.desc_dir + pred_class

        tranlator = googletrans.Translator()

        self.title_ko = pred_class.split('_')[-1]
        self.title_en = tranlator.translate(self.title_ko, dest='en', src='ko').text
        self.title_ja = tranlator.translate(self.title_ko, dest='ja', src='ko').text
        self.title_cn = tranlator.translate(self.title_ko, dest='zh-cn', src='ko').text
        