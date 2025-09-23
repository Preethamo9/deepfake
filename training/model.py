import torch, torch.nn as nn, torchvision.models as models, math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class ResNetTransformer(nn.Module):
    def __init__(self, num_classes=1, d_model=2048, nhead=8, num_encoder_layers=3, dim_feedforward=1024, freeze_resnet=True):
        super(ResNetTransformer, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        if freeze_resnet:
            for param in resnet.parameters(): param.requires_grad = False
        self.resnet_features = nn.Sequential(*list(resnet.children())[:-1])
        self.pos_encoder = PositionalEncoding(d_model=d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)
        self.classifier_video = nn.Sequential(nn.Linear(d_model, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, num_classes))
        self.classifier_image = nn.Sequential(nn.Linear(d_model, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, num_classes))
        self.d_model = d_model

    def forward(self, x):
        is_video = x.dim() == 5
        if is_video:
            batch_size, seq_length, c, h, w = x.shape
            x = x.view(batch_size * seq_length, c, h, w)
        with torch.no_grad() if next(self.resnet_features.parameters()).requires_grad is False else torch.enable_grad():
            features = self.resnet_features(x)
            features = features.view(features.size(0), -1)
        if is_video:
            features = features.view(batch_size, seq_length, self.d_model)
            features = self.pos_encoder(features)
            transformer_output = self.transformer_encoder(features)
            pooled_output = transformer_output.mean(dim=1)
            output = self.classifier_video(pooled_output)
        else:
            output = self.classifier_image(features)
        return output