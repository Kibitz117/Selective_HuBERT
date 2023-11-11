from transformers import HubertForSequenceClassification, HubertConfig
import torch

class HuBERTSelectiveNet(torch.nn.Module):
    def __init__(self, hubert_model, num_classes:int,feature_size:int, init_weights=True):
        super(HuBERTSelectiveNet, self).__init__()
        self.hubert_model = hubert_model
        self.dim_features = feature_size  # This should be 768 based on your config
        self.num_classes = num_classes
        
        # Classifier represented as f() in the original paper
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.dim_features, self.num_classes)
        )

        self.selector = torch.nn.Sequential(
            torch.nn.Linear(self.dim_features, self.dim_features),
            torch.nn.ReLU(True),
            # Normalize across the feature dimension, which is the last dimension of the input
            torch.nn.BatchNorm1d(self.dim_features), # self.dim_features should be 768
            torch.nn.Linear(self.dim_features, 1),
            torch.nn.Sigmoid()
        )

        # Auxiliary classifier represented as h() in the original paper
        self.aux_classifier = torch.nn.Sequential(
            torch.nn.Linear(self.dim_features, self.num_classes)
        )

        #Initialize weights of heads if required
        if init_weights:
            self._initialize_weights(self.classifier)
            self._initialize_weights(self.selector)
            self._initialize_weights(self.aux_classifier)

    def forward(self, input_values):
        # Run input through HuBERT model
        outputs = self.hubert_model(input_values)

        # Extract the last hidden state (features)
        x = outputs.last_hidden_state  # Extracts the tensor

        # Perform mean pooling over the timesteps
        # Assuming x has shape [batch_size, num_timesteps, num_features]
        x = torch.mean(x, dim=1)  # Now x has shape [batch_size, num_features]

        # Pass the pooled features through the classifier and selector heads
        prediction_out = self.classifier(x)
        selection_out = self.selector(x)
        auxiliary_out = self.aux_classifier(x)

        return prediction_out, selection_out, auxiliary_out

    def _initialize_weights(self, module):
        for m in module.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm1d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)
    def save_model(model, save_path, model_config):
        # Save the model state
        torch.save(model.state_dict(), save_path + "/model_state.pt")

        # Save the configuration
        with open(save_path + "/config.json", 'w') as f:
            json.dump(model_config, f)
    def load_model(load_path, hubert_model_class):
        # Load the configuration
        with open(load_path + "/config.json", 'r') as f:
            model_config = json.load(f)

        # Recreate the HuBERTSelectiveNet instance
        hubert_model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        model = hubert_model_class(hubert_model, model_config["num_classes"], model_config["feature_size"])

        # Load the model state
        model.load_state_dict(torch.load(load_path + "/model_state.pt"))
        model.eval()
        return model



