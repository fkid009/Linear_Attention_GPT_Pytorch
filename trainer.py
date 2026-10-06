import torch
from torch.nn import functional as F
import lightning as L


class LitGPT(L.LightningModule):
    def __init__(self, model, lr):
        super().__init__()
        self.model = model
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, batch):
        x, y = batch
        logits = self.model(x)  # (B, T, VOCAB_SIZE)
        return F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss  # 이 loss로 Lightning이 backward와 step을 해줌

    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log("val_loss", loss, prog_bar=True)  # val 전체 배치의 평균으로 기록됨

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
