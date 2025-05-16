import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.models.resnet import ResNet18_Weights
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False 

def convert_to_rgb(img):
    return img.convert('RGB')

def load_data(data_path='./data', batch_size=32):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        convert_to_rgb, 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    full_dataset = datasets.Caltech101(root=data_path, download=False, transform=transform)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, test_loader, len(full_dataset.categories)

def create_model():
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 101)
    return model

def evaluate(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = running_loss / total
    accuracy = correct / total
    
    return avg_loss, accuracy

def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs, device):
    train_loss_history = []
    test_loss_history = []
    train_acc_history = []
    test_acc_history = []
    
    best_test_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (images, labels) in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            progress_bar.set_description(f'Epoch [{epoch+1}/{num_epochs}]')
            progress_bar.set_postfix(loss=loss.item(), acc=correct/total)
        
        train_loss = running_loss / total
        train_acc = correct / total
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step(test_loss)

        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)
        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'best_model.pth')
        
        print(f'Epoch [{epoch+1}/{num_epochs}] - '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
    
    return {
        'train_loss': train_loss_history,
        'test_loss': test_loss_history,
        'train_acc': train_acc_history,
        'test_acc': test_acc_history,
        'best_test_acc': best_test_acc
    }

def plot_history(history, hyperparams, save_path='./training_plots'):
    os.makedirs(save_path, exist_ok=True)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='训练损失')
    plt.plot(history['test_loss'], label='测试损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.title(f'训练和测试损失曲线 (BS={hyperparams["batch_size"]}, LR={hyperparams["lr"]}, Best Test Acc={hyperparams["best_test_acc"]:.4f})')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='训练准确率')
    plt.plot(history['test_acc'], label='测试准确率')
    plt.xlabel('Epoch')
    plt.ylabel('准确率')
    plt.title(f'训练和测试准确率曲线 (BS={hyperparams["batch_size"]}, LR={hyperparams["lr"]}, Best Test Acc={hyperparams["best_test_acc"]:.4f})')
    plt.legend()
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/training_history_bs{hyperparams["batch_size"]}_lr{hyperparams["lr"]}_testAcc{hyperparams["best_test_acc"]}.png')
    plt.close()

def hyperparameter_search():

    hyperparam_grid = {
        'batch_size': [32, 64],
        'learning_rate': [0.001, 0.0005, 0.0001],
        'num_epochs': [10]
    }
    
    best_accuracy = 0.0
    best_params = None
    results = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    for batch_size in hyperparam_grid['batch_size']:
        train_loader, test_loader, num_classes = load_data(batch_size=batch_size)
        
        for lr in hyperparam_grid['learning_rate']:
            
            for epochs in hyperparam_grid['num_epochs']:
                print(f"\n=== 训练配置: Batch Size={batch_size}, Learning Rate={lr}")

                model = create_model().to(device)
                
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
                
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

                history = train_model(
                    model=model,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    criterion=criterion,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    num_epochs=epochs,
                    device=device
                )

                results.append({
                    'batch_size': batch_size,
                    'learning_rate': lr,
                    'num_epochs': epochs,
                    'best_test_accuracy': history['best_test_acc']
                })

                plot_history(history, {'batch_size': batch_size, 'lr': lr, 'best_test_acc': history['best_test_acc']})
                
                if history['best_test_acc'] > best_accuracy:
                    best_accuracy = history['best_test_acc']
                    best_params = {
                        'batch_size': batch_size,
                        'learning_rate': lr,
                        'num_epochs': epochs
                    }
                    torch.save(model.state_dict(), 'best_overall_model.pth')

    print("\n=== 超参数搜索结果 ===")
    for result in results:
        print(f"Batch Size={result['batch_size']}, LR={result['learning_rate']} "
              f"-> 最佳测试准确率: {result['best_test_accuracy']:.4f}")
    
    print(f"\n最佳参数组合: Batch Size={best_params['batch_size']}, "
          f"Learning Rate={best_params['learning_rate']}, "
          f"Epochs={best_params['num_epochs']}")
    print(f"最佳测试准确率: {best_accuracy:.4f}")

if __name__ == "__main__":
    hyperparameter_search()