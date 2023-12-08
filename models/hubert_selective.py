from transformers import HubertForSequenceClassification, HubertConfig
import torch


class HuBERTSelectiveNet(torch.nn.Module):
    def __init__(self, num_classes:int, feature_size:int, init_weights=True):
        super(HuBERTSelectiveNet, self).__init__()
        self.dim_features = feature_size  # This should be 768 based on your config
        self.num_classes = num_classes
        self.norm_after_activation = True 

        hidden_size = 1024

        #https://github.com/hearbenchmark/hear-eval-kit/blob/main/heareval/predictions/task_predictions.py
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.dim_features, hidden_size),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, self.num_classes)
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
            torch.nn.Linear(self.dim_features, hidden_size),
            torch.nn.Linear(hidden_size, self.num_classes),
            torch.nn.Sigmoid()
        )

        #Initialize weights of heads if required
        if init_weights:
            self._initialize_weights(self.classifier)
            self._initialize_weights(self.selector)
            self._initialize_weights(self.aux_classifier)

    def forward(self, input_values):
        prediction_out = self.classifier(input_values)
        selection_out = self.selector(input_values)
        auxiliary_out = self.aux_classifier(input_values)

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
        # TODO fix this it's wrong
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



