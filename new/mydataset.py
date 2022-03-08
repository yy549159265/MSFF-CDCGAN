from torchvision import transforms
from torch.utils.data import Dataset, DataLoader,random_split
from PIL import Image

# 定义读取文件的格式
def default_loader(path):
    return Image.open(path).convert('L')  # 以灰度图像读入

class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        super(MyDataset, self).__init__()
        fh = open(txt, 'r')
        imgs = []
        for line in fh.readlines():
            line = line.strip('\n')
            words = line.split("*")
            name = words[0].split('/')
            imgs.append((words[0], words[1], int(words[2]),name[4],words[3]),)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        SequenceImage, StructureImage, RNAlen,name,label = self.imgs[index]
        img1 = self.loader(SequenceImage)
        img2 = self.loader(StructureImage)
        if self.transform is not None:
            img1 = self.transform(img1)
        if self.target_transform is not None:
            img2 = self.target_transform(img2)
        return img1, img2, RNAlen,name,label

    def __len__(self):
        return len(self.imgs)

# transforms.ToTensor()把灰度范围从0-255变换到0-1之间,transform.Normalize()则把0-1变换到(-SeqImg,SeqImg)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
target_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])