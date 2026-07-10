import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import clip

class FakedditDataset(Dataset):
    def __init__( self, tsv_path, image_dir, preprocess):
        # load the TSV here
        self.df = pd.read_csv(tsv_path, sep = '\t')
        self.image_dir = image_dir
        self.preprocess = preprocess
    

    def __len__(self):
        # return number of samples
        return len(self.df)
    

    def __getitem__(self, idx):
        # return one (image, text, label) sample
        row = self.df.iloc[idx]
        text = row["clean_title"]
        label = row["2_way_label"]
        image_id = row["id"]

        for ext in [".jpg", ".png", ".jpeg"]:
            image_path = os.path.join(self.image_dir, f"{image_id}{ext}")
            if os.path.exists(image_path):
                break

        try:
            image = self.preprocess(Image.open(image_path).convert("RGB"))
        except:
            return self.__getitem__(idx+1)
        
        return image, text, label
    
def get_dataloader(tsv_path, image_dir, preprocess, batch_size=32, shuffle=True):
    dataset = FakedditDataset(tsv_path, image_dir, preprocess)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)