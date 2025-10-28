import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import pandas as pd
import time
from torch.utils.data import default_collate
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# --- 1. [과제 3] CIFAR-100 학습 ---

print()
print("====================CIFAR-100 VGG Training====================")

class VGG_Task3(nn.Module):
  def __init__(self):
    super(VGG_Task3, self).__init__()
    self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)
    self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size=3, stride = 1, padding=1)
    self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size=3, stride = 1, padding=1)
    self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 256, kernel_size=3, stride = 1, padding=1)
    self.conv4 = nn.Conv2d(in_channels= 256, out_channels= 256, kernel_size=3, stride = 1, padding=1)
    self.conv5 = nn.Conv2d(in_channels= 256, out_channels= 256, kernel_size=3, stride = 1, padding=1)
    self.relu = nn.ReLU(inplace = True)
    self.fc1 = nn.Linear(256 * 8 * 8, 4096) 
    self.fc2 = nn.Linear(4096, 100)

  def forward(self, x):
    x = self.relu(self.conv1(x))
    x = self.relu(self.conv2(x))
    x = self.maxpool(x)
    x = self.relu(self.conv3(x))
    x = self.relu(self.conv4(x))
    x = self.relu(self.conv5(x))
    x = self.maxpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc1(x)
    x = self.fc2(x)
    return x


def train_task3(epoch, model, train_loader, criterion, optimizer, num_epochs, total_step):
  model.train()
  for i, (images, labels) in enumerate(train_loader):
    images = images.to(device)
    labels = labels.to(device)

    outputs = model(images)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (i+1) % 100 == 0:
      print('Task 3 - Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))

def test_task3(model, test_loader):
  model.eval()
  with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
      images = images.to(device)
      labels = labels.to(device)
      outputs = model(images)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

  print('Task 3 - Accuracy of the model on the test images: {} %'.format(100 * correct / total))

cifar100_datapath = '/local_datasets/cifar_data_100'

transform_train_c100_simple = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()
])
transform_test_c100_simple = transforms.Compose([
    transforms.ToTensor()
])

train_dataset_c100 = torchvision.datasets.CIFAR100(
    root=cifar100_datapath, train=True, transform=transform_train_c100_simple, download=True
)
test_dataset_c100 = torchvision.datasets.CIFAR100(
    root=cifar100_datapath, train=False, transform=transform_test_c100_simple, download=True
)

train_loader_c100 = torch.utils.data.DataLoader(
    dataset=train_dataset_c100, batch_size=100, shuffle=True
)

dataiter_c100 = iter(train_loader_c100)
images_c100, _ = next(dataiter_c100)
mean_c100 = images_c100.mean(dim = [0, 2, 3])
std_c100 = images_c100.std(dim = [0, 2, 3])
print(f"CIFAR-100 Mean: {mean_c100}, Std: {std_c100}")

transform_train_c100 = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor(),
    transforms.Normalize(tuple(mean_c100), tuple(std_c100))
])
transform_test_c100 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(tuple(mean_c100), tuple(std_c100))
])

train_dataset_c100.transform = transform_train_c100
test_dataset_c100.transform = transform_test_c100

# 6. 최종 학습용 데이터로더 생성
train_loader_c100 = torch.utils.data.DataLoader(
    dataset=train_dataset_c100, batch_size=100, shuffle=True
)
test_loader_c100 = torch.utils.data.DataLoader(
    dataset=test_dataset_c100, batch_size=100, shuffle=False
)

# 과제 3 실행
num_epochs_task3 = 2
learning_rate_task3 = 0.001 # 원본 코드의 learning_rate 값 사용

model_task3 = VGG_Task3().to(device)
criterion_task3 = nn.CrossEntropyLoss()
optimizer_task3 = torch.optim.Adam(model_task3.parameters(), lr = learning_rate_task3)
total_step_task3 = len(train_loader_c100)

for epoch in range(num_epochs_task3):
  train_task3(epoch, model_task3, train_loader_c100, criterion_task3, optimizer_task3, num_epochs_task3, total_step_task3)
  test_task3(model_task3, test_loader_c100)

print("Task 3 (CIFAR-100) finished.")


# --- 2. [과제 1, 2, 4] CIFAR-10 하이퍼파라미터 튜닝 ---

print()
print("====================CIFAR-10 Hyperparameter Tuning====================")

cfg = {
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG_Custom_Test': [64, 'M', 128, 'M', 256, 'M']
}

class VGG(nn.Module):
    def __init__(self, vgg_cfg_name, pool_stride=2):
        super(VGG, self).__init__()
        self.features, last_channel = self._make_layers(cfg[vgg_cfg_name], pool_stride)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(last_channel, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _make_layers(self, cfg, pool_stride):
        layers = []
        in_channels = 3
        last_out_channel = in_channels
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=pool_stride)]
            else:
                last_out_channel = v
                layers += [
                    nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
                    nn.BatchNorm2d(v),
                    nn.ReLU(inplace=True),
                ]
                in_channels = v
        return nn.Sequential(*layers), last_out_channel


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def cutmix_data(x, y, alpha=1.0, num_classes=10):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        return x, y

    batch_size = x.size()[0]
    y_one_hot = F.one_hot(y, num_classes=num_classes).float().to(x.device)
    rand_index = torch.randperm(batch_size).to(x.device)
    y_a = y_one_hot
    y_b = y_one_hot[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x_cut = x.clone()
    x_cut[:, :, bby1:bby2, bbx1:bbx2] = x[rand_index, :, bby1:bby2, bbx1:bbx2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    new_y = y_a * lam + y_b * (1. - lam)
    return x_cut, new_y

NUM_CLASSES = 10 
CUTMIX_ALPHA = 1.0

def cutmix_collate_fn(batch):
    collated_batch = default_collate(batch)
    images, labels = collated_batch
    images, labels = cutmix_data(images, labels, 
                                 alpha=CUTMIX_ALPHA, 
                                 num_classes=NUM_CLASSES)
    return images, labels

def get_loaders(crop_size = 32, batch_size = 100, use_cutmix = False):
    
    dataset_path = '/local_datasets/cifar_data_10' 

    transform_train_simple = transforms.Compose([
        transforms.RandomCrop(crop_size, padding = 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test_simple = transforms.Compose([
        transforms.Resize(crop_size), 
        transforms.ToTensor(),
    ])

    train_dataset = torchvision.datasets.CIFAR10(root = dataset_path,
                                                 train = True,
                                                 download = True,
                                                 transform = transform_train_simple)
    
    test_dataset = torchvision.datasets.CIFAR10(root = dataset_path,
                                                train = False,
                                                download = True,
                                                transform = transform_test_simple)

    train_loader_for_norm = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size = 100,
                                                        shuffle=True)
    dataiter = iter(train_loader_for_norm)
    images, _ = next(dataiter)
    mean = images.mean(dim = [0, 2, 3])
    std = images.std(dim = [0, 2, 3])
    print(f"CIFAR-10 Mean : {mean}, Std: {std}")
    

    transform_train_with_norm = transforms.Compose([
        transforms.RandomCrop(crop_size, padding = 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(tuple(mean), tuple(std)),
    ])
    transform_test_with_norm = transforms.Compose([
        transforms.Resize(crop_size), 
        transforms.ToTensor(),
        transforms.Normalize(tuple(mean), tuple(std)),
    ])

    train_dataset.transform = transform_train_with_norm
    test_dataset.transform = transform_test_with_norm

    if use_cutmix:
        collate_fn_to_use = cutmix_collate_fn
    else:
        collate_fn_to_use = None

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size = batch_size,
                                               shuffle=True,
                                               collate_fn=collate_fn_to_use)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size = batch_size,
                                              shuffle = False)

    return train_loader, test_loader

def train_v_lib(model, train_loader, optimizer, criterion, device, epoch, num_epochs, total_step):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    start_time = time.time() 
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        if labels.dim() == 2:
            targets_for_acc = labels.argmax(dim=1)
        else:
            targets_for_acc = labels
        correct += predicted.eq(targets_for_acc).sum().item()
        if (i+1) % 100 == 0:
            print(f'Task 4 - Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')
    end_time = time.time()
    epoch_time = end_time - start_time
    epoch_acc = 100. * correct / total
    return epoch_acc, epoch_time 

def test_v_lib(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_acc = 100 * correct / total
    print(f'Task 4 - Accuracy of the model on the test images: {val_acc:.2f} %')
    return val_acc

results_df = pd.DataFrame(columns=['Experiment', 'Params', 'Total_Time_sec', 'Best_Val_Acc'])

def run_experiment(config, experiment_name):
    print()
    print(f"==================== Task 4 Sub-Experiment: [{experiment_name}] ====================")
    print(f"Config: {config}")
    
    train_loader, test_loader = get_loaders(
        crop_size=config['crop_size'], 
        batch_size=config['batch_size'],
        use_cutmix=config['use_cutmix'] 
    )
    
    model = VGG(vgg_cfg_name=config['model_cfg'], 
                pool_stride=config['pool_stride']).to(device)
    
    param_count = count_parameters(model)
    print(f"파라미터 수: {param_count:,}") 

    if config['use_cutmix']:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    else:
        criterion = nn.CrossEntropyLoss()
        
    optimizer = optim.SGD(model.parameters(), 
                          lr=config['learning_rate'], 
                          momentum=0.9, 
                          weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])

    total_train_time = 0
    best_val_acc = 0.0
    total_step = len(train_loader)
    
    for epoch in range(config['epochs']):
        train_acc, epoch_time = train_v_lib(
            model, train_loader, optimizer, criterion, device,
            epoch=epoch, 
            num_epochs=config['epochs'], 
            total_step=total_step
        )
        val_acc = test_v_lib(model, test_loader, device)
        scheduler.step()
        total_train_time += epoch_time
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
    print(f"--- Best Val Acc: {best_val_acc:.2f}% ---")
    
    global results_df
    new_row = {'Experiment': experiment_name, 
               'Params': param_count, 
               'Total_Time_sec': total_train_time, 
               'Best_Val_Acc': best_val_acc}
    results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)

BASE_EPOCHS = 2
BASE_LR = 0.01
BASE_BATCH_SIZE = 128
BASE_MODEL_CFG = 'VGG_Custom_Test'

experiment_configs = [
    { 'name': 'Baseline (Crop 32, Stride 2)',
      'config': { 'model_cfg': BASE_MODEL_CFG, 'pool_stride': 2, 'crop_size': 32,
                  'batch_size': BASE_BATCH_SIZE, 'use_cutmix': False,
                  'epochs': BASE_EPOCHS, 'learning_rate': BASE_LR } },
    { 'name': 'Struct: Pool Stride 1',
      'config': { 'model_cfg': BASE_MODEL_CFG, 'pool_stride': 1, 'crop_size': 32,
                  'batch_size': BASE_BATCH_SIZE, 'use_cutmix': False,
                  'epochs': BASE_EPOCHS, 'learning_rate': BASE_LR } },
    { 'name': 'Preproc: RandomCrop 24',
      'config': { 'model_cfg': BASE_MODEL_CFG, 'pool_stride': 2, 'crop_size': 24,
                  'batch_size': BASE_BATCH_SIZE, 'use_cutmix': False,
                  'epochs': BASE_EPOCHS, 'learning_rate': BASE_LR } },
    { 'name': 'Preproc: RandomCrop 28',
      'config': { 'model_cfg': BASE_MODEL_CFG, 'pool_stride': 2, 'crop_size': 28,
                  'batch_size': BASE_BATCH_SIZE, 'use_cutmix': False,
                  'epochs': BASE_EPOCHS, 'learning_rate': BASE_LR } },
    { 'name': 'Method: CutMix (ON) + Crop 32',
      'config': { 'model_cfg': BASE_MODEL_CFG, 'pool_stride': 2, 'crop_size': 32,
                  'batch_size': BASE_BATCH_SIZE, 'use_cutmix': True,
                  'epochs': BASE_EPOCHS, 'learning_rate': BASE_LR } },
    { 'name': 'Method: CutMix (ON) + Crop 28',
      'config': { 'model_cfg': BASE_MODEL_CFG, 'pool_stride': 2, 'crop_size': 28,
                  'batch_size': BASE_BATCH_SIZE, 'use_cutmix': True,
                  'epochs': BASE_EPOCHS, 'learning_rate': BASE_LR }
    }
]

# 과제 4 실행
for exp in experiment_configs:
    run_experiment(config=exp['config'], experiment_name=exp['name'])

print()
print("==================== 모든 실험 종료 ====================")

# 과제 4 결과 CSV 파일로 저장
output_filename = 'lab7_experiment_results.csv'
results_df.to_csv(output_filename, index=False)
print(f"Task 4 (CIFAR-10) 결과를 {output_filename} 에 저장")