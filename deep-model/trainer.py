from datetime import datetime

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

class Trainer:
    def __init__(self, inputs ,model, criterion, optimizer, device):
        self.inputs = inputs
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def measure_diff(self, logits, y):
        _, predicted_labels = torch.max(logits, 1)
        true_logits = logits[range(logits.shape[0]), y]
        predicted_logits = logits[range(logits.shape[0]), predicted_labels]
        diff = torch.abs(true_logits - predicted_logits)
        return diff.mean()

    def train(self, train_loader, num_epochs=20):
        self.model.train()
        total_loss, total_diff = [], []

        for epoch in range(num_epochs):
            epoch_loss, epoch_diff = 0.0, 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                epoch_diff += self.measure_diff(outputs, labels).item()

            total_loss.append(epoch_loss / len(train_loader))
            total_diff.append(epoch_diff / len(train_loader))
            print(f'Epoch {epoch + 1}, Loss: {total_loss[-1]}, Diff: {total_diff[-1]}')

        # visualize loss and diff
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(np.arange(num_epochs), total_loss, label='Train Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Epoch vs Loss')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(np.arange(num_epochs), total_diff, label='Prediction Diff')
        plt.xlabel('Epochs')
        plt.ylabel('Difference')
        plt.title('Epoch vs Difference')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('loss.png', dpi=300)
        plt.show()

        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        model_path = f"{self.inputs.model_name}_{current_time}.pt"
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved at: {model_path}")

    def evaluate(self, test_loader):
        self.model.eval()
        total_loss, total_diff = 0.0, 0.0
        all_predicted, all_labels = [], []

        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                all_predicted.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                total_loss += loss.item()
                total_diff += self.measure_diff(outputs, labels).item()

        avg_loss = total_loss / len(test_loader)
        avg_diff = total_diff / len(test_loader)
        print(f'Evaluation - Loss: {avg_loss}, Diff: {avg_diff}')

        return np.array(all_predicted), np.array(all_labels)

    def visualize_predictions(self,predictions, all_labels):
        plt.figure(figsize=(8, 8))
        plt.scatter(all_labels, predictions, alpha=0.5)
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('True vs Predicted Labels')
        plt.grid(True)
        plt.savefig(self.inputs.model_name + '_scatter.png', dpi=300)
        plt.show()

    def visualize_confusion_matrix(self,all_predicted, all_labels,report_file='classification_report.txt'):
        cm = confusion_matrix(all_labels, all_predicted)
        plt.figure(figsize=(10, 10))
        sns.heatmap(cm, annot=True, fmt=".0f", square=True, cmap='Blues')
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(self.inputs.model_name + '_confusion_matrix.png',dpi=300)
        plt.show()

        report_file = self.inputs.model_name + '_classification_report.txt'
        with open(report_file, 'w') as f:
            print("\nClassification Report:", file=f)
            print(classification_report(all_labels, all_predicted), file=f)