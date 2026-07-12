def preprocess(data_folder: str) -> tuple[pd.DataFrame, dict]:
    #path object pointing to root data directory
    data_dir = Path(data_folder)

    #empty label dict
    label = {}
    folder_names = [f.name for f in data_dir.iterdir() if f.is_dir()]#list comprehension method for getting foldernames

    for count, name in enumerate(folder_names):#assigning labels to folder names
        label[name] = count

    image_data = []
    for image_path in data_dir.rglob("*.jpg"):#getting all images
        row_entry = {
            "folder": image_path.parent.name,
            "file_name": image_path.name,
            "label" : label[image_path.parent.name]
        }
        image_data.append(row_entry)

    df = pd.DataFrame(image_data)
    return df,label

class SatelliteDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = row["label"]
        img = Image.open(Path("data")/row["folder"]/row["file_name"])
        if self.transform:
            img = self.transform(img)
        return img,label
    
#defining data loader for the training data
set_seed(42)
data_train_transform = transforms.Compose([
    transforms.Resize((64, 64)), #Resize images for CNN training
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomHorizontalFlip(p=0.5),#some data Augumentation
    transforms.RandomRotation(180),#data augumentation
    transforms.RandomApply(     # apply jitter/blur with 50% probability
        [transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
         transforms.GaussianBlur(3, sigma=(0.1, 2.0))],
        p=0.5
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])          
])
data_transform = transforms.Compose([
    transforms.Resize((64, 64)), #Resize images for CNN validation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])          
])
train_dataset = SatelliteDataset(df=train_df, transform=data_train_transform)
val_dataset = SatelliteDataset(df=val_df, transform=data_transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# your code goes here
from copy import deepcopy

loss_func = nn.CrossEntropyLoss(label_smoothing=0.05)#defining loss function


def num_correct(pred, y):
    pred = pred.argmax(1)# get the predicted class
    correct = (pred == y).sum()# compare with the true labels
    return correct.item()# return the number of correct items



def train(train_loader, val_loader, model, loss_fn, optimizer):

    size = len(train_loader.dataset)
    model.train()                                      # set the model to training mode

    train_loss = 0
    train_correct = 0

    for batch, (X, y) in enumerate(train_loader):  # loop over the training data batches
        X, y = X.to(device), y.to(device)              # move data to GPU if available

        pred = model(X)                                # forward pass (compute the predictions)
        loss = loss_fn(pred, y)                        # compute loss

        train_loss += loss.item() * len(y)             # accumulate loss (weighted for batch size)
        train_correct += num_correct(pred, y)          # accumulate number of correct items

        optimizer.zero_grad()                          # reset the gradients
        loss.backward()                                # backpropagation (compute the gradients)
        optimizer.step()                               # update the parameters
        scheduler.step()

        if batch % 100 == 0:
            loss_value = loss.item()
            current = batch * len(X)
            batch_acc = num_correct(pred, y) / len(y)
            print(f"loss: {loss_value:>7f} [{current:>5d}/{size:>5d}], acc: {batch_acc*100:.2f}%")

    train_loss /= size
    train_acc = train_correct / size

    print(f"\nTraining loss after epoch: {train_loss:>7f}, acc: {train_acc*100:.2f}%")

    # validation
    size = len(val_loader.dataset)
    model.eval()                                       # set the model to evaluation mode
                                                       # (no dropout, no batch normalization, etc.)
    val_loss = 0
    val_correct = 0

    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = loss_fn(pred, y)

            val_loss += loss.item() * len(y)
            val_correct += num_correct(pred, y)

    val_loss /= size
    val_acc = val_correct / size

    print(f"Validation loss: {val_loss:>7f}, acc: {val_acc*100:.2f}%\n")

    return train_loss, train_acc, val_loss, val_acc


best_model_params = None
best_val_acc = None
epoch_with_best_val_acc = 0

model = SatelliteCNN().to(device)      # fresh model instance
epochs = 150
optimizer = AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)#update model param based on gradients
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2)

early_stopping = 20                 # stop training if the validation loss doesn't improve for this many epochs

t_loss = []
t_acc=[]
v_loss = []
v_acc=[]

for t in range(epochs):
    print(f"Epoch {t + 1}\n----------------------------------------")
    tl, ta, vl, va = train(train_loader, val_loader, model, loss_func, optimizer)
    t_loss.append(tl)
    t_acc.append(ta)
    v_loss.append(vl)
    v_acc.append(va)
    print(f"Epoch {t + 1} done.\n")
    print()
    
    if best_val_acc is None or va > best_val_acc:
        best_val_acc = va
        best_model_params = deepcopy(model.state_dict())
        epoch_with_best_val_acc = t

    if t > 0 and t - epoch_with_best_val_acc >= early_stopping:
        print(f"Early stopping at epoch {t + 1}.\n")
        break
print("Training done.")

model.load_state_dict(best_model_params)
        
save_file = Path("assets/weights")
save_file.mkdir(parents=True, exist_ok=True)
save_path = save_file/"best_model.pth"
torch.save(best_model_params, save_path)