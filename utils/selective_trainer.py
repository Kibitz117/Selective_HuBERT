import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers import HubertModel, AutoConfig
import sys
from tqdm import tqdm
#import Adam
from transformers.optimization import AdamW
sys.path.append('utils')
from selective_loss import SelectiveLoss
sys.path.append('models/')
from hubert_selective import HuBERTSelectiveNet
def main():
    # Load the HuBERT model
    epochs = 5
    pretrained_model_name = "facebook/hubert-base-ls960"
    hubert_model = HubertModel.from_pretrained(pretrained_model_name)

    # Rest of your parameters
    num_classes = 2  # Define the number of classes for your classification task
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    # Load the data
    mfccs = np.load("data/queen_and_no_queen_waveforms.npy")
    labels=np.load("data/queen_and_no_queen_labels.npy")  # You need to have your labels ready

    # Convert to torch tensors
    mfccs_tensor = torch.tensor(mfccs, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    # Create the dataset
    dataset = TensorDataset(mfccs_tensor, labels_tensor)

    # Split the dataset into training and testing
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    #Get feature size of train
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = hubert_model(inputs)
        feature_size=outputs.last_hidden_state.shape[2]
        break
     # Create an instance of your custom HuBERTSelectiveNet
    model = HuBERTSelectiveNet(hubert_model, num_classes=num_classes,feature_size=feature_size) # Move model to the appropriate device

    loss_func = torch.nn.CrossEntropyLoss()
    coverage = 0.8
    alpha = 0.5
    lm = 32.0

    selective_loss = SelectiveLoss(loss_func, coverage, alpha, lm)
    
    # loss=loss
    # Create the optimizer
    optimizer = AdamW(model.parameters(), lr=1e-5)
    # Train the model
    model.train()
    for epoch in tqdm(range(epochs)):
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero out the gradients
            optimizer.zero_grad()

            # Forward pass
            logits, selection_logits, auxiliary_logits = model(inputs)

            #Mean aux out to fit cross entropy loss
            auxiliary_logits=auxiliary_logits.mean(dim=1)
            #convert labels to long
            #labels=labels.long()
            #Average the timesteps
            loss_dict = selective_loss(prediction_out=logits,
                                    selection_out=selection_logits,
                                    auxiliary_out=auxiliary_logits,
                                    target=labels)

            # Extract the loss value for backpropagation
            loss = loss_dict['loss']  # Assuming 'loss' is the correct key for the main loss value
            loss.backward()

            # Clip the gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update the parameters
            optimizer.step()

    # Save the model
    model_config = {
    "num_classes": num_classes,  # Adjust as per your model
    "feature_size": feature_size  # Adjust as per your model
}

    # # Save the model after training
    # model.save_model(model, "", model_config)

    model.eval()
    total_eval_accuracy = 0
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            logits, selection_logits, auxiliary_logits = model(inputs)

        # Calculate the accuracy for this batch of test sentences
        total_eval_accuracy += flat_accuracy(logits, labels)

    # Report the final accuracy for this test run
    avg_test_accuracy = total_eval_accuracy / len(test_loader)
    print("Accuracy: {0:.2f}".format(avg_test_accuracy))

def flat_accuracy(preds, labels):
    pred_flat = torch.argmax(preds, dim=1)  # Returns indices of maximum values along the class dimension
    correct_predictions = pred_flat == labels  # Element-wise comparison
    accuracy = torch.sum(correct_predictions).item() / len(labels)  # Calculate accuracy
    return accuracy

if __name__ == "__main__":
    main()
