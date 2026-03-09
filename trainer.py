import torch
from utils import accuracy, save_checkpoint

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, criterion, device, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.config = config # 왜 이게 필요힌지??

    def train_epoch(self):
        self.model.train()
        total_loss, total_acc = 0.0, 0.0
        
        for image, label in self.train_loader:
            image, label = image.to(self.device), label.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(image)
            loss = self.criterion(output, label)
            loss.backward()

            #gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            total_acc += accuracy(output, label)

        avg_loss = total_loss/len(self.train_loader)
        avg_acc = total_acc/len(self.train_loader)
        return avg_loss, avg_acc
    
    def validate(self):
        self.model.eval()
        total_loss, total_acc = 0.0, 0.0

        with torch.no_grad():
            for image, label in self.val_loader:
                image, label = image.to(self.device), label.to(self.device)

                output = self.model(image)
                loss = self.criterion(output, label)

                total_loss += loss.item()
                total_acc += accuracy(output, label)

            avg_loss = total_loss/len(self.val_loader)
            avg_acc = total_acc/len(self.val_loader)
            return avg_loss, avg_acc
        
    def best(self):
        best_val_acc = 0.0

        for epoch in range(self.config.epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            print(f"Epoch [{epoch+1}/{self.config.epochs}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_checkpoint(self.model, self.optimizer, epoch, val_acc, self.config.save_path)
                print(f"  → Best model saved (val_acc: {best_val_acc:.4f})")





