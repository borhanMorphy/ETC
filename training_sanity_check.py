from typing import List

from tqdm import tqdm
from dataset import DummyDataset

from src import ETC, config, VanillaTransformer

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class ETCClassifier(nn.Module):
    def __init__(self, c, in_features: int, num_classes: int):
        super().__init__()
        self.etc = ETC(c)
        self.cls_head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        _, z_global = self.etc(x)
        # z_global: B x Sg x d

        return self.cls_head(z_global)


class Tokenizer:
    def __init__(
        self,
        max_length: int,
        num_aux_tokens: int = 0,
        cls_token_idx: int = None,
    ):
        self._vocab = [
            "-",
            "p",
            "1",
            "b",
            "v",
            "s",
            "l",
            "o",
            "j",
            "h",
            "y",
            "4",
            ".",
            "g",
            " ",
            "|",
            "/",
            "a",
            ",",
            "0",
            "2",
            "7",
            "6",
            "5",
            "c",
            "3",
            "m",
            "f",
            "t",
            "d",
            "e",
            "r",
            "u",
            "8",
            "n",
            "9",
            "i",
        ]
        self._pad_token_idx = 0
        self._vocab_to_idx = {
            ch: i + 1 + num_aux_tokens for i, ch in enumerate(self._vocab)
        }
        self._num_aux_tokens = num_aux_tokens
        self.max_length = max_length
        self.cls_token_idx = cls_token_idx

    @property
    def vocab_size(self) -> int:
        return len(self._vocab) + self._num_aux_tokens + 1

    def __call__(self, text: str) -> List[int]:
        if self.cls_token_idx is None:
            token_ids = []
        else:
            token_ids = [self.cls_token_idx]

        for ch in text:
            token_ids.append(self._vocab_to_idx[ch.lower()])

        token_ids = token_ids[: self.max_length]

        num_pads = self.max_length - len(token_ids)

        token_ids = token_ids + [self._pad_token_idx] * num_pads

        assert len(token_ids) == self.max_length

        return token_ids

    @property
    def padding_idx(self) -> int:
        return self._pad_token_idx


if __name__ == "__main__":
    MAX_SEQ_LEN = 2**6
    num_aux_tokens = 1
    cls_token_idx = None

    tokenizer = Tokenizer(
        max_length=MAX_SEQ_LEN,
        num_aux_tokens=num_aux_tokens,
        cls_token_idx=cls_token_idx,
    )

    train_ds = DummyDataset(num_samples=2000, callback=tokenizer)

    train_dl = DataLoader(train_ds, batch_size=32)

    batch = next(iter(train_dl))

    model_config = config.ModelConfig(
        d_model=64,
        num_heads=4,
        dim_feedforward=2 * 64,
        dropout=0.0,
        num_layers=2,
        num_classes=-1,  # TODO
        vocab_size=tokenizer.vocab_size,
        padding_idx=tokenizer.padding_idx,
        long_to_global_ratio=16,
        l2l=config.ETCAttentionConfig(
            rel_pos_max_distance=16,
            local_attention_radius=16,
            attention_type="sparse",
            directed_relative_position=True,
        ),
        l2g=config.ETCAttentionConfig(
            rel_pos_max_distance=8,
            local_attention_radius=None,
            attention_type="dense",
            directed_relative_position=False,
        ),
        g2g=config.ETCAttentionConfig(
            rel_pos_max_distance=4,
            local_attention_radius=None,
            attention_type="dense",
            directed_relative_position=True,
        ),
        g2l=config.ETCAttentionConfig(
            rel_pos_max_distance=8,
            local_attention_radius=None,
            attention_type="dense",
            directed_relative_position=False,
        ),
    )

    device = "cuda"

    # needs CLS Token
    """
    model = VanillaTransformer(
        model_config.d_model,
        model_config.num_heads,
        tokenizer.vocab_size,
        MAX_SEQ_LEN,
        num_layers=2,
    )
    """

    # """
    model = ETCClassifier(
        model_config,
        model_config.d_model * (MAX_SEQ_LEN // model_config.long_to_global_ratio),
        1,
    )
    # """

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("number of parameters: ", num_params)

    model.to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
    # """
    for epoch in range(200):
        total_loss = []
        for batch in tqdm(train_dl):
            optimizer.zero_grad()

            logits = model.forward(batch["token_ids"].to(device))
            # logits: B x 1

            loss = criterion(logits, batch["label"].to(device))

            loss.backward()
            optimizer.step()

            total_loss.append(loss.item())

        print(
            "epoch {} loss -> {}".format(epoch + 1, sum(total_loss) / len(total_loss))
        )
    # """
    # """
    model.eval()
    for batch in tqdm(train_dl):
        with torch.no_grad():
            logits = model.forward(batch["token_ids"].to(device))

            scores = logits.sigmoid()

            for score, label, dt1, dt2 in zip(
                scores, batch["label"].to(device), batch["date_1"], batch["date_2"]
            ):
                print(dt1)
                print(dt2)
                print(score, label)
                input("\n")
    # """
