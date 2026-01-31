# Mô hình CRNN để đọc chữ trong vùng text
import timm
import torch.nn as nn
import torch


class CRNN(nn.Module):
    def __init__(
        self, vocab_size, hidden_size, n_layers, dropout=0.2, unfreeze_layers=3
    ):
        super(CRNN, self).__init__()

        # ------------------------------------
        # 1️⃣ Backbone CNN: ResNet34 từ timm
        # ------------------------------------
        # in_chans=1: ảnh đầu vào là grayscale (1 kênh)
        # pretrained=True: dùng trọng số pretrain trên ImageNet
        backbone = timm.create_model("resnet34", in_chans=1, pretrained=True)

        # Lấy tất cả các layer trừ 2 layer cuối (adaptive pool + FC)
        modules = list(backbone.children())[:-2]
        modules.append(nn.AdaptiveAvgPool2d((1, None)))

         # Gộp thành backbone CNN hoàn chỉnh
        self.backbone = nn.Sequential(*modules)

        # Unfreeze the last few layers
        for parameter in self.backbone[-unfreeze_layers:].parameters():
            parameter.requires_grad = True

        self.mapSeq = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Dropout(dropout))

        self.gru = nn.GRU(
            512,
            hidden_size,
            n_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.layer_norm = nn.LayerNorm(hidden_size * 2)

        self.out = nn.Sequential(
            nn.Linear(hidden_size * 2, vocab_size), nn.LogSoftmax(dim=2)
        )

    @torch.autocast(device_type="cuda")
    def forward(self, x):
        # x: (B, C=1, H, W)

        # 1️⃣ Trích xuất đặc trưng bằng CNN
        x = self.backbone(x)          # → (B, 512, 1, W')

        # 2️⃣ Hoán vị trục để coi chiều rộng W' là chiều time-step
        x = x.permute(0, 3, 1, 2)     # → (B, W', 512, 1)

        # 3️⃣ Flatten chiều (C, H) = (512, 1) thành 512
        x = x.view(x.size(0), x.size(1), -1)  # → (B, W', 512)

        # 4️⃣ Qua mapSeq (Linear + ReLU + Dropout)
        x = self.mapSeq(x)            # → (B, W', 512)

        # 5️⃣ Qua GRU hai chiều
        x, _ = self.gru(x)            # → (B, W', hidden_size*2)

        # 6️⃣ LayerNorm để ổn định training
        x = self.layer_norm(x)        # → (B, W', hidden_size*2)

        # 7️⃣ Qua head Linear + LogSoftmax
        x = self.out(x)               # → (B, W', vocab_size)

        # 8️⃣ Đổi sang dạng (T, B, C) cho CTC Loss
        x = x.permute(1, 0, 2)        # → (W', B, vocab_size)

        return x                      # T = W' (sequence length), B = batch size