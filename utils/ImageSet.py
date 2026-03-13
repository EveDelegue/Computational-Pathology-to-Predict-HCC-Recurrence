from torch.utils.data import Dataset

class ImageSet(Dataset):
    def __init__(self, data, labels, transforms):
        self.X=data
        self.y=labels
        self.transforms=transforms

    def __len__(self):
        return len(self.y)

    def __getitem__(self,idx):
        im1,im2,im3=self.X[idx]
        imgs=[self.transforms(im) for im in [im1,im2,im3]]
        label=self.y[idx]
        return imgs,label