from typing import List

from tqdm import tqdm
from dataset import DummyDataset

from src import ETC, ModelConfig, VanillaTransformer

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Tokenizer():
    def __init__(self, max_length: int):
        self._vocab = ['-', 'p', '1', 'b', 'v', 's', 'l', 'o', 'j', 'h', 'y', '4', '.', 'g', ' ', '|', '/', 'a', ',', '0', '2', '7', '6', '5', 'c', '3', 'm', 'f', 't', 'd', 'e', 'r', 'u', '8', 'n', '9', 'i']
        self._pad_token_idx = 0
        self._start_token = 1
        self._vocab_to_idx = {ch: i + 2 for i, ch in enumerate(self._vocab)}
        self.max_length = max_length

    @property
    def vocab_size(self) -> int:
        return len(self._vocab) + 2

    def __call__(self, text: str) -> List[int]:
        token_ids = [self._start_token]

        for ch in text:
            token_ids.append(
                self._vocab_to_idx[ch.lower()]
            )

        token_ids = token_ids[:self.max_length]

        num_pads = self.max_length - len(token_ids)

        token_ids = token_ids + [self._pad_token_idx] * num_pads

        assert len(token_ids) == self.max_length

        return token_ids

if __name__ == '__main__':
    MAX_SEQ_LEN = 2**6

    tokenizer = Tokenizer(max_length=MAX_SEQ_LEN)

    train_ds = DummyDataset(num_samples=2000, callback=tokenizer)

    train_dl = DataLoader(train_ds, batch_size=32)

    config = ModelConfig(
        num_features=64,
        num_heads=2,
        vocab_size=tokenizer.vocab_size,
        sliding_window_radius=5,
        segment_radius=1,
        hard_masking=True,
        global_token_ratio=4,
    )
    device = "cuda"

    
    model = VanillaTransformer(
        config.num_features,
        config.num_heads,
        tokenizer.vocab_size,
        MAX_SEQ_LEN,
        num_layers=2,
    )
    """

    model = ETC(
        config.num_features,
        config.num_heads,
        tokenizer.vocab_size,
        config.sliding_window_radius,
        config.segment_radius,
        hard_masking=config.hard_masking,
        global_token_ratio=config.global_token_ratio,
        num_of_global_token_types=1,
        num_layers=1,
    )
    """

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("number of parameters: ", num_params)

    model.to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
    #"""
    for epoch in range(100):
        total_loss = []
        for batch in tqdm(train_dl):
            optimizer.zero_grad()

            logits = model.forward(batch["token_ids"].to(device))

            loss = criterion(logits, batch["label"].to(device))

            loss.backward()
            optimizer.step()

            total_loss.append(loss.item())

        print("epoch {} loss -> {}".format(epoch+1, sum(total_loss) / len(total_loss)))
    #"""
    #"""
    model.eval()
    for batch in tqdm(train_dl):
        with torch.no_grad():
            logits = model.forward(batch["token_ids"].to(device))

            scores = logits.sigmoid()

            for score, label, dt1, dt2 in zip(scores, batch["label"].to(device), batch["date_1"], batch["date_2"]):
                print(dt1)
                print(dt2)
                print(score, label)
                input("\n")
    #"""