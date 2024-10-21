# train.py

# This file contains the main training loop.
# This file should include:

# 1. Set up configuration
#    - Parse command-line arguments or load config file
#    - Set hyperparameters (learning rate, batch size, epochs, etc.)
#    - Configure device (CPU/GPU)

# 2. Model initialization
#    - Initialize the PortraitNet model
#    - Move model to appropriate device (CPU/GPU)

# 3. Training loop
#    - Iterate through epochs

# 4. Model evaluation
#    - Evaluate final model performance 

# 5. Save trained model
#    - Save final model weights and architecture


# Usage:
# python train.py --config path/to/config.yaml
import torch.optim as optim
import torch
from FacialNet import *
from utils import *
from dataset import *
import time
# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initialize model and loss function
Fmodel = FacialNet().to(device)
# criterion = TotalLoss()  # use the loss function we defined before
criterion = F_total_loss()
optimizer = optim.Adam(Fmodel.parameters(), lr=0.001)

# load the previous trained model (if exists)
checkpoint_path = 'checkpoints/FacialNet.pth'
if os.path.exists(checkpoint_path):
    Fmodel.load_state_dict(torch.load(checkpoint_path))
    print(f"Continuing training")
else:
    # if the checkpoint is not exists, initialize the model parameters
    for name, param in Fmodel.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass
# training loop
def train_model_portrait(model, train_loader, val_loader, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, image_hat, masks, edges in train_loader:
            images, image_hat, masks, edges = images.to(device), image_hat.to(device), masks.to(device), edges.to(device)
            # forward propagation
            optimizer.zero_grad()
            output_mask, output_edge  = model(images)
            output_mask_hat, _ = model(image_hat)
            loss = criterion(output_mask, output_edge, masks, edges, output_mask, output_mask_hat)  # calculate the loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
        # validate the model
        if val_loader is not None:  
            val_iou = validate_model(model, val_loader)
            print(f'IoU: {val_iou:.4f}')

def train_model_face(model, train_loader, val_loader, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, image_hat, masks in train_loader:
            images, image_hat, masks = images.to(device), image_hat.to(device), masks.to(device)
            # forward propagation
            optimizer.zero_grad()
            output_mask = model(images)
            output_mask_hat = model(image_hat)
            loss = criterion(output_mask, masks, output_mask, output_mask_hat)  # calculate the loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

        # validate the model
        if val_loader is not None:  
            val_iou = validate_model_face(model, val_loader)
            print(f'IoU: {val_iou:.4f}')



# validate the model
def validate_model_face(model, val_loader):
    model.eval()
    running_iou = 0.0
    with torch.no_grad():
        for images, _, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            running_iou += compute_iou_face(outputs, masks) * images.size(0)
    val_iou = running_iou / len(val_loader.dataset)
    return val_iou


# validate the model
def validate_model(model, val_loader):
    model.eval()
    running_iou = 0.0
    with torch.no_grad():
        for images, _, masks, _ in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs, _ = model(images)
            running_iou += compute_iou(outputs, masks) * images.size(0)
    val_iou = running_iou / len(val_loader.dataset)
    return val_iou

# train the model
image_dir_EG = 'EG1800/Images'
mask_dir_EG = 'EG1800/Labels'
train_file_list_EG = 'EG1800/eg1800_train.txt'
val_file_list_EG = 'EG1800/eg1800_test.txt'
root_dir_MH = 'matting_human_sample'
root_dir_EP = 'EasyPortrait'



train_dataset_EP = EasyPortraitDataset(root_dir=root_dir_EP, split='train', transform=True)
val_dataset_EP = EasyPortraitDataset(root_dir=root_dir_EP, split='val', transform=False)

print(f'Train dataset size: {len(train_dataset_EP)}')
print(f'Val dataset size: {len(val_dataset_EP)}')

train_loader_EP = DataLoader(train_dataset_EP, batch_size=64, shuffle=True, num_workers=8)
val_loader_EP = DataLoader(val_dataset_EP, batch_size=64, shuffle=False, num_workers=8)

start_time = time.time()
train_model_face(Fmodel, train_loader_EP, val_loader_EP, criterion, optimizer, num_epochs=200)
torch.save(Fmodel.state_dict(), 'FacialNet.pth')
end_time = time.time()
print(f'Training time: {end_time - start_time:.2f}s')