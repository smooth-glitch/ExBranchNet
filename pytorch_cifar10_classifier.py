import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

# ------------------------------
# 1. Data Loading and Preprocessing
# ------------------------------
def get_dataloaders(batch_size=128):
    """
    Download CIFAR-10 and create DataLoaders for training and test sets.
    Applies data augmentation on the training set.
    """
    # Transformations for training: random flip, random crop, normalization.
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.247, 0.243, 0.261))
    ])

    # Transformations for testing: only normalization.
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.247, 0.243, 0.261))
    ])

    # Download and create CIFAR-10 datasets.
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    # Create DataLoaders.
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return trainloader, testloader

# ------------------------------
# 2. Model Architecture Definition
# ------------------------------
class Stem(nn.Module):
    """
    Stem: Extracts low-level features using a convolution, BatchNorm, and ReLU.
    """
    def __init__(self, in_channels=3, out_channels=64):
        super(Stem, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Block(nn.Module):
    """
    Backbone Block:
    - Expert branch: Uses global average pooling and two FC layers (with ReLU and Softmax) to compute weights.
    - K convolution branches: Each branch processes the input; their outputs are weighted and summed.
    """
    def __init__(self, channels, num_experts, reduction_ratio):
        super(Block, self).__init__()
        self.num_experts = num_experts
        
        # Expert branch components.
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction_ratio)
        self.fc2 = nn.Linear(channels // reduction_ratio, num_experts)
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)
        
        # Create K convolution branches.
        self.convs = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
            for _ in range(num_experts)
        ])
    
    def forward(self, x):
        B, C, H, W = x.size()
        # Expert branch: global average pooling and FC layers.
        pooled = self.avgpool(x).view(B, C)
        fc_out = self.relu(self.fc1(pooled))
        fc_out = self.fc2(fc_out)
        weights = self.softmax(fc_out)  # Shape: [B, num_experts]
        
        # Process x through each convolution branch.
        conv_outputs = [conv(x) for conv in self.convs]
        conv_stack = torch.stack(conv_outputs, dim=1)  # Shape: [B, num_experts, C, H, W]
        
        # Reshape weights for broadcasting and compute weighted sum.
        weights = weights.view(B, self.num_experts, 1, 1, 1)
        out = (conv_stack * weights).sum(dim=1)
        return out

class Backbone(nn.Module):
    """
    Backbone: A sequence of N blocks that progressively refines features.
    """
    def __init__(self, num_blocks, channels, num_experts, reduction_ratio):
        super(Backbone, self).__init__()
        self.blocks = nn.Sequential(
            *[Block(channels, num_experts, reduction_ratio) for _ in range(num_blocks)]
        )
    
    def forward(self, x):
        return self.blocks(x)

class Classifier(nn.Module):
    """
    Classifier: Maps the final feature map to class logits.
    Applies BatchNorm, global average pooling, and a fully connected layer.
    """
    def __init__(self, in_channels, num_classes):
        super(Classifier, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, num_classes)
    
    def forward(self, x):
        x = self.bn(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor.
        return self.fc(x)

class Model(nn.Module):
    """
    Complete Model: Combines Stem, Backbone, and Classifier as specified.
    """
    def __init__(self, in_channels=3, stem_channels=64, num_blocks=3, num_experts=4, reduction_ratio=4, num_classes=10):
        super(Model, self).__init__()
        self.stem = Stem(in_channels, stem_channels)
        self.backbone = Backbone(num_blocks, stem_channels, num_experts, reduction_ratio)
        self.classifier = Classifier(stem_channels, num_classes)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.backbone(x)
        return self.classifier(x)

# ------------------------------
# 3. Training Pipeline
# ------------------------------
def train_model(model, trainloader, testloader, device, epochs=20):
    """
    Train the model using SGD with momentum and a learning rate scheduler.
    Logs training loss, training accuracy, and validation accuracy.
    (Reduced to 20 epochs for faster execution on non-NVIDIA GPUs.)
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    # With 20 epochs, adjust milestones (e.g., reduce LR at epoch 10 and 15)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15], gamma=0.1)
    
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()
        
        scheduler.step()
        
        epoch_loss = running_loss / total_train
        epoch_train_acc = 100.0 * correct_train / total_train
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_train_acc)
        
        # Evaluate on test set.
        model.eval()
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total_test += labels.size(0)
                correct_test += predicted.eq(labels).sum().item()
        epoch_test_acc = 100.0 * correct_test / total_test
        test_accuracies.append(epoch_test_acc)
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% | Test Acc: {epoch_test_acc:.2f}%")
    
    return train_losses, train_accuracies, test_accuracies

def plot_curves(train_losses, train_accuracies, test_accuracies):
    """
    Plot training loss and accuracy curves using Matplotlib.
    """
    epochs_range = range(1, len(train_losses)+1)
    plt.figure(figsize=(12,5))
    
    # Plot training loss.
    plt.subplot(1,2,1)
    plt.plot(epochs_range, train_losses, 'o-', label='Training Loss')
    plt.title("Training Loss Evolution")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    # Plot training and validation accuracy.
    plt.subplot(1,2,2)
    plt.plot(epochs_range, train_accuracies, 'o-', label='Training Accuracy')
    plt.plot(epochs_range, test_accuracies, 'o-', label='Validation Accuracy')
    plt.title("Accuracy Evolution")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# ------------------------------
# 4. Main Function
# ------------------------------
def main():
    batch_size = 128
    epochs = 20  # Reduced number of epochs for faster execution
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data.
    trainloader, testloader = get_dataloaders(batch_size=batch_size)
    
    # Create the model.
    model = Model().to(device)
    
    # Train the model.
    train_losses, train_accuracies, test_accuracies = train_model(model, trainloader, testloader, device, epochs=epochs)
    
    # Plot training curves.
    plot_curves(train_losses, train_accuracies, test_accuracies)
    
    # Save the trained model.
    torch.save(model.state_dict(), "cifar10_model.pth")
    print("Model saved as 'cifar10_model.pth'.")

if __name__ == "__main__":
    main()
