import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import DatasetFolder, ImageFolder
from torchvision.datasets import DatasetFolder, ImageFolder
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS
from torchvision import transforms
import os

transform = transforms.Compose([
    transforms.ToTensor(),  # convert image to PyTorch tensor
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # normalize images
])


class CustomImageFolder(datasets.DatasetFolder):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.classes.sort(key=int)  # sorts classes numerically

        self.imgs = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                self.imgs.append((os.path.join(class_dir, img_name), int(class_name)))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path, label = self.imgs[idx]
        img = default_loader(img_path)
        if self.transform:
            img = self.transform(img)
        return img, label



def visualize_images_batch(dataset, loader, batch_size=9):
    images, labels = next(iter(loader))
    sqrt_batch_size = int(np.sqrt(batch_size))
    fig = plt.figure(figsize=(8, 8))
    for i in range(batch_size):
        ax = fig.add_subplot(sqrt_batch_size, sqrt_batch_size, i + 1, xticks=[], yticks=[])  # adjust layout (4x4 here) based on batch size
        ax.imshow(np.transpose(images[i].numpy(), (1, 2, 0)))  # remove the normalization and permutation
        ax.set_title(dataset.classes[labels[i]])
        # add a white border around the images
        [i.set_linewidth(3) for i in ax.spines.values()]
        [i.set_color('white') for i in ax.spines.values()]

    plt.subplots_adjust(wspace=0.0001)  # adjust the white space between the plots
    plt.show(dpi=300)

def visualize_labels_dist(dataset):
    # Count occurrences of each label
    label_count = {}
    for _, label in dataset.imgs:
        label_count[label] = label_count.get(label, 0) + 1

    # Generate label and count lists
    labels = list(label_count.keys())
    counts = list(label_count.values())

    # Calculate percentages
    total = sum(counts)
    percentages = [100 * count / total for count in counts]

    # Build new labels including label number, count, and percentage
    labels = [f'{k}-{v} ({p:.1f}%)' for k, v, p in zip(labels, counts, percentages)]

    # Generate pie chart
    fig, ax = plt.subplots()
    wedges, _ = ax.pie(counts, pctdistance=0.85)

    # Draw a circle at the center (for aesthetics)
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig.gca().add_artist(centre_circle)

    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')

    # Create legend and place it in the upper right corner
    ax.legend(wedges, labels,fontsize=6, title="Labels", loc="upper right", bbox_to_anchor=(1.3, 1))
    plt.title('Objects Distribution')
    plt.tight_layout()
    plt.show()

def create_data_set(folder_path, batch_size, visualize_batch=False):
    dataset = CustomImageFolder(root_dir=folder_path, transform=transform)
    print("Following classes are there: ",len(dataset.classes))
    print("There are total {} images".format(len(dataset)))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    visualize_labels_dist(dataset)
    if visualize_batch is True:
        visualize_images_batch(dataset, dataloader, batch_size)
    # calculate split sizes
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader