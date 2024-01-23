from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, Subset
from sklearn.model_selection import KFold, StratifiedKFold
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import os
import torchvision
import PIL
from sklearn.model_selection import train_test_split
from collections import Counter
import cv2
from torch.utils.tensorboard import SummaryWriter


RANDOM_SEED = 43

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def load_attacked_database(path, batch_size, folder_from, image_size=(128,128)): 
        tf_image = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # ######### Validation ###############
        val_data = datasets.ImageFolder(os.path.join(path, folder_from), transform=tf_image) 
        
        num_class = len(os.listdir(os.path.join(path, folder_from)))

        print("Database report: \n")
        val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)
        print(val_data)

        return val_loader, num_class

def load_database(path, batch_size, image_size=(128,128), is_agumentation=False):
        if is_agumentation:
            tf_image_train = transforms.Compose([#transforms.ToPILImage(),
                                       transforms.Resize(image_size),
                                       #transforms.AutoAugment(transforms.autoaugment.AutoAugmentPolicy.CIFAR10),
                                       #transforms.RandomRotation(degrees=30),
                                       #transforms.RandomResizedCrop(224, scale=(0.8, 1.2)),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomVerticalFlip(),
                                       #transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=15),
                                       transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    ])
            tf_image_test = transforms.Compose([
                transforms.Resize(image_size),
                #transforms.AutoAugment(transforms.autoaugment.AutoAugmentPolicy.CIFAR10),
                transforms.ToTensor(),
                #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            tf_image_train = tf_image_test = transforms.Compose([
                transforms.Resize(image_size),
                #transforms.AutoAugment(transforms.autoaugment.AutoAugmentPolicy.CIFAR10),
                transforms.ToTensor(),
                #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        
        ######### Train ###############
        train_data = datasets.ImageFolder(os.path.join(path, 'Train'), transform=tf_image_train)

        ######### test ###############
        test_data = datasets.ImageFolder(os.path.join(path, 'Test'), transform=tf_image_test)

        # ######### Validation ###############
        # val_data = datasets.ImageFolder(os.path.join(path, 'val'), transform=tf_image) 
        
        num_class = len(os.listdir(os.path.join(path, 'Train')))

        print("Database report: \n")
        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=4)
        print(train_data)

        test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=4)
        print(test_data)

        # val_loader = thutils.data.DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)
        # print(val_data)

        return train_loader, test_loader, num_class

def load_images_path(img_dir, image_size = (128, 128)):
        tf_image = transforms.Compose([transforms.ToTensor(),  
                                       transforms.Resize(image_size), 
                                       transforms.AutoAugment(transforms.autoaugment.AutoAugmentPolicy.CIFAR10),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    ])

        database = ImageFolder(root=img_dir, transform=tf_image)
        
        return database

def load_database_kf(root_path, batch_size, image_size=(128,128), csv_path=None, is_agumentation=False, n_folds=5, as_rgb=False):
        if is_agumentation:
            tf_image = transforms.Compose([#transforms.ToPILImage(),
                                       transforms.Resize(image_size),
                                       #transforms.AutoAugment(transforms.autoaugment.AutoAugmentPolicy.CIFAR10),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomVerticalFlip(),
                                       transforms.RandomRotation(30),
                                    #    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                                    #    transforms.RandomResizedCrop(224),
                                       transforms.ToTensor(),
                                       #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    ])
        else:
            tf_image = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
        #kf = StratifiedKFold(n_splits=n_folds)
        train_loader, test_loader = {}, {}
        #database = None
        num_class = 0
        if csv_path is None:
            database = datasets.ImageFolder(root_path, transform=tf_image)
            num_class = len(os.listdir(root_path))
        else:
            database = CustomDatasetFromCSV(path_root=root_path, tf_image=tf_image, csv_name=csv_path, as_rgb=as_rgb)
            num_class = len(database.cl_name.values())
        
        for i, (train_index, test_index) in enumerate(kf.split(database)):
                
                #train = Subset(database, train_index)
                train_sampler = SubsetRandomSampler(train_index)
                test_sampler = SubsetRandomSampler(test_index)
                #idx = int(len(test_index)*0.1)
                #test = Subset(database, test_index)
                #val_sampler = SubsetRandomSampler(test_index[0:idx])
                
                train_loader[i] = DataLoader(database, batch_size=batch_size, sampler=train_sampler, num_workers=4)
                test_loader[i] = DataLoader(database, batch_size=batch_size, sampler=test_sampler, num_workers=4)
                #val_loader[i] = DataLoader(database, batch_size=batch_size, sampler=val_sampler, num_workers=4)

        return train_loader, test_loader, num_class

def load_database_df(root_path, csv_path, batch_size, image_size=(128,128), is_agumentation=False, test_size=None, as_rgb=False):
        if is_agumentation:
            tf_image = transforms.Compose([#transforms.ToPILImage(),
                                       transforms.Resize(image_size),
                                       #transforms.AutoAugment(transforms.autoaugment.AutoAugmentPolicy.CIFAR10),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomVerticalFlip(),
                                       #transforms.RandomAffine(degrees=3, shear=0.01),
                                       #transforms.RandomResizedCrop(size=image_size, scale=(0.875, 1.0)),
                                       #transforms.ColorJitter(brightness=(0.7, 1.5)),
                                       transforms.ToTensor(),
                                       #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    ])
        else:
            tf_image = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        
        if test_size is None:
            train = CustomDatasetFromCSV(root_path, tf_image=tf_image, csv_name=csv_path, task="Train", as_rgb=as_rgb)
            test = CustomDatasetFromCSV(root_path, tf_image=tf_image, csv_name=csv_path, task="Test", as_rgb=as_rgb)
            num_class = len(train.cl_name.values())
            
            print(train.cl_name)
            
            train_loader = DataLoader(train, batch_size=batch_size, num_workers=4, shuffle=True)
            test_loader = DataLoader(test, batch_size=batch_size, num_workers=4, shuffle=False)
        else:
            data = CustomDatasetFromCSV(root_path, tf_image=tf_image, csv_name=csv_path, as_rgb=as_rgb)
            
            print({k: cl for k, cl in enumerate(data.cl_name)})
            
            train, test = train_test_split(list(range(len(data))), test_size=test_size, shuffle=True, random_state=RANDOM_SEED)
            
            # index_num = int(np.floor(0.1*len(test)))
            # test_index = test[:len(test)-index_num]
            
            train_sampler = SubsetRandomSampler(train)
            test_sampler = SubsetRandomSampler(test)
            
            num_class = len(data.cl_name.values())
            
            train_loader = DataLoader(data, batch_size=batch_size, sampler=train_sampler, num_workers=4)
            test_loader = DataLoader(data, batch_size=batch_size, sampler=test_sampler, num_workers=4)
            
            #print(Counter(train_loader.dataset))

        return train_loader, test_loader, num_class

def get_random_images(root_path, csv_path, batch_size, image_size, n_samples=100, set="Test"):
        tf_image = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        val = CustomDatasetFromCSV(root_path, tf_image=tf_image, csv_name=csv_path, task=set)
            
        val_index = np.random.choice(range(len(val)), size=n_samples, replace=False)
            
        sampler_val = Subset(val, val_index)
            
        num_class = len(val.cl_name.values())
            
        random_loader = DataLoader(sampler_val, batch_size=batch_size, num_workers=4, shuffle=False)
        
        return random_loader

  
def load_attacked_database_df(root_path, csv_path, batch_size, image_size=(128,128), percentage_attacked=0.1, test_size=None):
        tf_image = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        if test_size is None:
            val = CustomDatasetFromCSV(root_path, tf_image=tf_image, csv_name=csv_path, task="Val")
            
            val_index = np.random.choice(range(len(val)), size=100, replace=False)
            
            sampler_val = Subset(val, val_index)
            
            num_class = len(val.cl_name.values())
            
            val_loader = DataLoader(sampler_val, batch_size=batch_size, num_workers=4, shuffle=False)
        else:
            data = CustomDatasetFromCSV(root_path, tf_image=tf_image, csv_name=csv_path)
            
            train, test = train_test_split(list(range(len(data))), test_size=test_size, random_state=RANDOM_SEED)
            
            num_class = len(data.cl_name.values())
            
            index_num = int(np.floor(percentage_attacked*len(test)))
            val_index = test[len(test)-index_num:]
            
            sampler_val = Subset(data, val_index)
            
            val_loader = DataLoader(sampler_val, batch_size=batch_size, num_workers=4, shuffle=False)

        return val_loader, num_class

def show_images(dataset_loader, db_name, path_to_save):
    """function that show images from dataloader

    Args:
        dataset_loader (_type_): _description_
        db_name (_type_): _description_

    """
    os.makedirs(path_to_save, exist_ok=True)
    batch = next(iter(dataset_loader))
    images, labels = batch
        
    plt.figure(figsize=(11, 11))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(make_grid(images[:32], padding=2, normalize=True), (1, 2, 0)))
    #plt.savefig("./attack-images/preview_train_{}.png".format(db_name), bbox_inches='tight', pad_inches=0)
    plt.savefig(os.path.join(path_to_save, "preview_train_{}.png".format(db_name)))

def show_images_from_array(images_array, db_name):
    """function that show images from dataloader

    Args:
        dataset_loader (_type_): _description_
        db_name (_type_): _description_

    """        
    plt.figure(figsize=(11, 11))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(make_grid(torch.Tensor(images_array[:32]), padding=2, normalize=True), (1, 2, 0)))
    plt.savefig("./attack-images/preview_train_{}.png".format(db_name), bbox_inches='tight', pad_inches=0)
    
def show_random_adv_image(images_array, db_name, attack_name, eps, path_to_save):
    
    image_idx = np.random.randint(0, len(images_array))
    
    os.makedirs(path_to_save, exist_ok=True)
    
    plt.figure(figsize=(11, 11))
    plt.axis("off")
    plt.imshow(np.transpose(make_grid(torch.Tensor(images_array), normalize=True), (1, 2, 0)))
    plt.savefig(os.path.join(path_to_save,"attack_preview_{}_{}_{}.png".format(db_name, attack_name, eps)), bbox_inches='tight', pad_inches=0)

def save_random_train_images(train_loader, experiment_name, dataset_name, one_channel=False):
        
        writer = SummaryWriter('../metrics/{}'.format(experiment_name))
        
        # get some random training images
        dataiter = iter(train_loader)
        images, labels = next(dataiter)

        # create grid of images
        img_grid = make_grid(images)

        # show images
        if one_channel:
            img = img.mean(dim=0)
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        if one_channel:
            plt.imshow(npimg, cmap="Greys")
        else:
            plt.imshow(np.transpose(npimg, (1, 2, 0)))

        # write to tensorboard
        writer.add_image(dataset_name, img_grid)
    
def select_n_random(data, labels, classes, experiment_name, n=100):
        '''
        Selects n random datapoints and their corresponding labels from a dataset
        '''
        assert len(data) == len(labels)
        
        writer = SummaryWriter('../metrics/{}'.format(experiment_name))

        perm = torch.randperm(len(data))
        images_random  = data[perm][:n] 
        label_random = labels[perm][:n]

        # get the class labels for each image
        class_labels = [classes[lab] for lab in label_random]

        # log embeddings
        features = images_random.view(-1, 28 * 28)
        writer.add_embedding(features,
                            metadata=class_labels,
                            label_img=images_random.unsqueeze(1))
        writer.close() 
    
def save_all_adv_image(path_to_save, images_array, labels, db_name , attack_name, cls=["akiec","bcc","bkl","df","mel","nv","vasc"]):
    
    if not os.path.exists(os.path.join(path_to_save, db_name, attack_name)):
        os.mkdir(os.path.join(path_to_save, db_name, attack_name))
    
    for c in cls:
        if not os.path.exists(os.path.join(path_to_save, db_name, attack_name, c)):
            os.mkdir(os.path.join(path_to_save, db_name, attack_name, c))
    
    plt.figure(figsize=(11, 11))
    plt.axis("off")
    for i in range(len(images_array)):
        plt.axis("off")
        plt.imshow(np.transpose(make_grid(torch.Tensor(images_array[i]), normalize=True), (1, 2, 0)))
        plt.savefig("{}/{}/{}/{}/{}_{}_{}.png".format(path_to_save, db_name, attack_name, cls[int(labels[i])], cls[int(labels[i])], attack_name, i), bbox_inches='tight', pad_inches=0)
    
def read_model_from_checkpoint(model_path, model_name, nb_class):

    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    state_dict = {key[6:] : checkpoint['state_dict'][key] for key in checkpoint['state_dict']}
    model = __get_model_structure(model_name, nb_class)
    model.load_state_dict(state_dict)
    
    return model

def __get_model_structure(model_name, nb_class):
    model = None
    #"resnet50" "vgg16" "vgg19" "inceptionv3" "densenet" "efficientnet"
    nb_class = nb_class if nb_class > 2 else 1
    if model_name == "resnet50":
        model = torchvision.models.resnet50()
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Sequential(
                torch.nn.Linear(num_ftrs, 224),
                torch.nn.BatchNorm1d(224),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(224, nb_class)
        )

    elif model_name == "vgg16":
        model = torchvision.models.vgg.vgg16()
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = torch.nn.Sequential(
                torch.nn.Linear(num_ftrs, 128),
                torch.nn.BatchNorm1d(128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(128, nb_class),
        )
    
    elif model_name == "vgg19":
        model = torchvision.models.vgg.vgg19()
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = torch.nn.Sequential(
                torch.nn.Linear(num_ftrs, 128),
                torch.nn.BatchNorm1d(128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(128, nb_class),
        )
    
    elif model_name == "inceptionv3":
        model = torchvision.models.inception_v3()
        model.aux_logits = False
        
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Sequential(
                torch.nn.Linear(num_ftrs, 128),
                torch.nn.BatchNorm1d(128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(128, nb_class),
        )       
    
    elif model_name == "efficientnet":
        model = torchvision.models.efficientnet_b0()
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Sequential(
                torch.nn.Linear(num_ftrs, 128),
                torch.nn.BatchNorm1d(128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(128, nb_class),
        )
    
    elif model_name == "densenet":
        model = torchvision.models.densenet121()
        num_ftrs = model.classifier.in_features
        model.classifier = torch.nn.Sequential(
                torch.nn.Linear(num_ftrs, 128),
                torch.nn.BatchNorm1d(128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(128, nb_class),
        )
    
    return model

def numpy_to_dataloader(images, labels, batch_size):
    
    dataset  = CustomDataset(images, labels)
    
    val_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    
    return val_loader

def dataloader_to_numpy(dataloader):
    
    images, labels = zip(*[dataloader.dataset[i] for i in range(len(dataloader.dataset))])
    images = torch.stack(images).numpy() 
    labels = np.array(labels)
    
    return images, labels

class CustomDatasetFromCSV(Dataset):
    def __init__(self, path_root, tf_image, csv_name, as_rgb=False, task=None):
        self.data = pd.read_csv(csv_name)
        self.as_rgb = as_rgb
        if task is not None:
            self.data.query("Task == @task", inplace=True)
        self.tf_image = tf_image
        self.root = path_root
        self.cl_name = {c: i for i, c in enumerate(np.unique(self.data["y"]))}
        self.BARVALUE = "/" if not os.name == "nt" else "\\"
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        #x_path = os.path.join(self.root, self.data.iloc[idx, 0].split(self.BARVALUE)[-2], self.data.iloc[idx, 0].split(self.BARVALUE)[-1])
        x_path = os.path.join(self.root, self.data.iloc[idx, 0])
        y = self.cl_name[self.data.iloc[idx, 1]]
        
        X = Image.open(x_path).convert("RGB")
        #X = cv2.cvtColor(cv2.imread(x_path), cv2.COLOR_BGR2RGB) if self.as_rgb else cv2.imread(x_path, cv2.IMREAD_GRAYSCALE)
 
        if self.tf_image:
            X = self.tf_image(X)
        
        return X, y
    
class CustomDataset(Dataset):
    
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        
        X = self.images[idx]
        y = self.labels[idx]
        
        # if self.transform:
        #     x = Image.fromarray(self.data[idx].transpose(1,2,0))
        #     x = self.transform(x)
        
        return X, y

class DatasetFromFolder(Dataset):
    def __init__(self, img_dir, image_size=(128,128)):
        tf_image = transforms.Compose([ transforms.Resize(image_size),
                                        transforms.ToTensor(),  
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    ])
        self.dataset = ImageFolder(root=img_dir, transform=tf_image)
        self.X = []
        self.y = []
        
        self.X, self.y = zip(*[(x, y) for x, y in self.dataset])
        
        # for img, y in self.dataset:
        #     self.X.append(img)
        #     self.y.append(y)
        
        self.X = torch.stack(list(self.X))
        self.y = torch.tensor(list(self.y))
        
        #self.y = self.y.unsqueeze(-1)
        
        # print('Shape of x:', self.X.shape)
        # print('Shape of y:', self.y.shape)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        
        return x, y