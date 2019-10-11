import torch
import torchvision.transforms as transforms
from PIL import Image


class FileManager:
    def __init__(self, content_image_path, style_image_path,
                 device, max_shape):
        self.content_image_path = content_image_path
        self.style_image_path = style_image_path
        self.device = device
        self.max_shape = max_shape

    def read_images(self):
        content_image = Image.open(self.content_image_path).convert('RGB')
        style_image = Image.open(self.style_image_path).convert('RGB')
        height, width = self.get_shape(content_image)
        loader = transforms.Compose([
            transforms.Resize((height, width)),
            transforms.ToTensor()])
        content_image = loader(content_image)[:3, :, :].unsqueeze(0)
        style_image = loader(style_image)[:3, :, :].unsqueeze(0)
        return content_image.to(self.device, torch.float), \
               style_image.to(self.device, torch.float)

    def get_shape(self, content_image):
        if self.max_shape[0] < content_image.size[0]:
            new_height = content_image.size[1] / (content_image.size[0] / self.max_shape[0])
            return int(new_height), self.max_shape[0]
        elif self.max_shape[1] < content_image.size[1]:
            new_width = content_image.size[0] / (content_image.size[1] / self.max_shape[1])
            return self.max_shape[1], int(new_width)
        else:
            return content_image[1], content_image[0]


    @staticmethod
    def save_image(image, output_path):
        FileManager.reverse_transform(image).save(output_path)

    @staticmethod
    def reverse_transform(image):
        unloader = transforms.ToPILImage()  # reconvert into PIL image
        image = image.cpu().clone()
        image = image.squeeze(0)
        image = unloader(image)
        return image
