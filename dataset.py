from torchvision import transforms
from torch.utils.data import Dataset
import os
from PIL import Image


class Datasets(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.dataset = os.listdir(data_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.data_path, self.dataset[index]))
        image_data = self.trans(image)
        return image_data


if __name__ == '__main__':
    data = Datasets(r"C:\sample\Cartoon_faces\faces")
    image_data = data[0]
    print(image_data)
    print(image_data.shape)
