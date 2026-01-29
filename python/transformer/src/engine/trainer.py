import numpy as np
import torch
import copy


def validation(model, valid_loader, criterion):
    model.eval()
    avg_valid_loss = 0

    with torch.no_grad():
        for x_val, y_val in valid_loader:
            outputs = model(x_val)
            loss = criterion(outputs, y_val)

            avg_valid_loss += loss.item()

    avg_valid_loss /= len(valid_loader)
    return avg_valid_loss
    

def train(model, train_loader, valid_loader, criterion, optimizer, num_epochs = 100):

    train_hist = np.zeros(num_epochs)
    valid_hist = np.zeros(num_epochs)
    
    best_val_loss = float('inf')
    best_epoch = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        model.train()
        avg_train_loss = 0
        
        for x_train, y_train in train_loader:
            outputs = model(x_train)
            loss = criterion(outputs, y_train)   
                             
            optimizer.zero_grad()
            loss.backward()
            
            # gradient 폭발을 막기 위해 추가
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            
            optimizer.step()
            
            avg_train_loss += loss.item()
               
        avg_train_loss /= len(train_loader)
        train_hist[epoch] = avg_train_loss
        
        avg_valid_loss = validation(model, valid_loader, criterion)
        valid_hist[epoch] = avg_valid_loss
        
        if avg_valid_loss < best_val_loss:
            best_val_loss = avg_valid_loss
            best_epoch = epoch + 1
            best_model_wts = copy.deepcopy(model.state_dict()) 
        
        print(
            f'Epoch [{epoch+1:03d}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Valid Loss: {avg_valid_loss:.4f}'
        )
    
    model.load_state_dict(best_model_wts)
    print(f'✅ Best model saved at epoch {best_epoch} (val loss = {best_val_loss:.4f})')
            
    return model.eval(), train_hist, valid_hist