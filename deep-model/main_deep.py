import torch
from torch import nn

import parser
import pickle
import warnings
import dataset
import model
import trainer
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    parsed_args = parser.Parser()
    inputs = parsed_args.parse()
    data_folder = './ORGANIZED-DATA-NO-BLUE/1'
    train_loader, test_loader = dataset.create_data_set(data_folder, inputs.batch,True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.Net_arc().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=inputs.learning_rate, momentum=0.9)
    trainer = trainer.Trainer(inputs, model, criterion, optimizer, device)
    trainer.train(train_loader, num_epochs=1000)
    all_predicted, all_labels = trainer.evaluate(test_loader)
    trainer.visualize_predictions(all_predicted, all_labels)
    trainer.visualize_confusion_matrix(all_predicted, all_labels)