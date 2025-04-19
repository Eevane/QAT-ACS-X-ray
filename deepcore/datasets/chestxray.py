from torchvision import transforms
from torch.utils.data import Dataset
import torch
import pandas as pd
import os
from PIL import Image
from sklearn.model_selection import train_test_split

class NIHChestXrayDataset(Dataset):
    def __init__(self, img_dir, dataframe, transform=None):
        self.img_dir = img_dir
        self.data = dataframe  # input Dataframe

        self.transform = transform

        # 14 labels, all 0 for No Finding
        self.class_names = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
            'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation',
            'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
        ]
        self.class_to_idx = {label: i for i, label in enumerate(self.class_names)}

        self.classes = self.class_names

        # fallback
        #self.targets = self.data['Finding Labels'].apply(
        #    lambda s: self._to_main_class_index(s)
        #).tolist()

    def _to_main_class_index(self, label_str):
        if label_str == 'No Finding':
            return -1  
        for label in label_str.split('|'):
            if label in self.class_to_idx:
                return self.class_to_idx[label]
        return -1


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path_png = os.path.join(self.img_dir, row['Image Index'])
        img_path, _ = os.path.splitext(img_path_png)
        img_path_jpg = img_path + '.jpg'

        # load image
        try:
            image = Image.open(img_path_jpg).convert("RGB")
        except Exception as e:
            print(f'error{e} when loading image')

        # initialize 0 vector
        label_vector = torch.zeros(len(self.class_names), dtype=torch.float32)

        # label to one-hot
        labels_str = row['Finding Labels']
        if labels_str != 'No Finding':
            labels = labels_str.split('|')
            for label in labels:
                if label in self.class_to_idx:
                    label_vector[self.class_to_idx[label]] = 1.0

        if self.transform:
            image = self.transform(image)

        return image, label_vector

    
def chestxray(data_path, test_size=0.2, random_state=42):
    channel = 3
    im_size = (224, 224)
    num_classes = 14
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.Resize(im_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    img_dir = os.path.join(data_path, "images_processed_002")
    label_csv = os.path.join(data_path, "Data_Entry_2017.csv")  

    #full_df = pd.read_csv(label_csv)
    full_df = pd.read_csv(label_csv)[['Image Index', 'Finding Labels']]

    # folder 2 only
    start_name = "00001336_000.png"
    end_name = "00003923_013.png"
    start_idx = full_df[full_df['Image Index'] == start_name].index[0]
    end_idx = full_df[full_df['Image Index'] == end_name].index[0]
    subset_df = full_df.iloc[start_idx:end_idx+1].reset_index(drop=True)

    train_df, test_df = train_test_split(
        subset_df, test_size=test_size, random_state=random_state, shuffle=True
    )

    # .copy()
    train_dataset = NIHChestXrayDataset(img_dir, train_df.copy(), transform)
    test_dataset = NIHChestXrayDataset(img_dir, test_df.copy(), transform)

    class_names = train_dataset.class_names

    return channel, im_size, num_classes, class_names, mean, std, train_dataset, test_dataset
