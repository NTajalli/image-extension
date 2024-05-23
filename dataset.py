import os
from torch.utils.data import Dataset
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, input_dir, output_dir, transform=None):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.transform = transform
        self.input_images = sorted(os.listdir(input_dir))
        self.output_images = sorted(os.listdir(output_dir))

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        input_image = Image.open(os.path.join(self.input_dir, self.input_images[idx]))
        output_image = Image.open(os.path.join(self.output_dir, self.output_images[idx]))

        if self.transform:
            input_image = self.transform(input_image)
            output_image = self.transform(output_image)

        return {'input': input_image, 'real': output_image}
