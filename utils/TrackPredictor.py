import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple

class Encoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 2):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # x shape: (batch_size, seq_len, input_size)
        outputs, (hidden, cell) = self.lstm(x)
        return outputs, (hidden, cell)

class Decoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 2):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # x shape: (batch_size, 1, input_size)
        output, (hidden, cell) = self.lstm(x, hidden)
        prediction = self.fc(output)
        return prediction, (hidden, cell)

class Seq2Seq(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 2):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, num_layers)
        self.decoder = Decoder(input_size, hidden_size, output_size, num_layers)

    def forward(self, src: torch.Tensor, trg: torch.Tensor, teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        # src shape: (batch_size, src_seq_len, input_size)
        # trg shape: (batch_size, trg_seq_len, input_size)
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.fc.out_features

        # 存储输出
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(src.device)

        # 编码器前向传播
        encoder_outputs, (hidden, cell) = self.encoder(src)

        # 解码器的第一个输入是编码器的最后一个输出
        input = src[:, -1:, :]

        for t in range(trg_len):
            # 解码器前向传播
            output, (hidden, cell) = self.decoder(input, (hidden, cell))

            # 存储预测
            outputs[:, t:t+1] = output

            # 决定是否使用教师强制
            teacher_force = np.random.random() < teacher_forcing_ratio

            # 下一个输入是真实值或预测值
            input = trg[:, t:t+1] if teacher_force else output

        return outputs

class TrackPredictor:
    def __init__(self, input_size: int = 2, hidden_size: int = 64, output_size: int = 2, num_layers: int = 2):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Seq2Seq(input_size, hidden_size, output_size, num_layers).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def train_epoch(self, train_data: List[np.ndarray], seq_len: int = 10, pred_len: int = 5) -> float:
        self.model.train()
        total_loss = 0

        for sequence in train_data:
            # 准备输入和目标序列
            src = torch.FloatTensor(sequence[:-pred_len]).to(self.device)
            trg = torch.FloatTensor(sequence[-pred_len:]).to(self.device)

            # 添加批次维度
            src = src.unsqueeze(0)
            trg = trg.unsqueeze(0)

            # 前向传播
            output = self.model(src, trg)

            # 计算损失
            loss = self.criterion(output, trg)

            # 反向传播和优化
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_data)

    def predict(self, sequence: np.ndarray, pred_len: int = 5) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            src = torch.FloatTensor(sequence).to(self.device)
            src = src.unsqueeze(0)

            # 创建目标序列的占位符
            trg = torch.zeros(1, pred_len, src.shape[2]).to(self.device)

            # 预测
            output = self.model(src, trg, teacher_forcing_ratio=0)

            return output.squeeze(0).cpu().numpy()

    def save_model(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load_model(self, path: str):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])