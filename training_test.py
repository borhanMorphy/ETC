from typing import List

from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DistilBertTokenizer, DistilBertModel

from src import ETC, ModelConfig, VanillaTransformer

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Tokenizer():
    def __init__(self):
        self._vocab = ['ó', 'j', 'e', '#', 'q', 'ï', 'o', '‘', 'õ', '|', 'ä', '+', '¢', '«', '&', 'í', 'k', '{', '\x91', '-', '½', 'ö', 'é', 'ë', 'y', '¨', '§', '\x84', 'è', '\x96', '8', 'ù', '6', '³', '0', 'ø', '\x08', '₤', '`', '\x9a', 'a', '>', '?', '\xa0', '£', '^', '4', 'ü', 'ý', 't', '\x8e', '1', 'ð', '7', '¡', '…', '°', '\x10', '(', '2', '~', 'à', 'û', '"', '\x80', "'", '\x97', 'm', '”', 'º', 'ñ', '$', '’', 'p', '\x9e', 'f', '“', '·', 'á', 'c', 'v', 'z', '¾', 'â', 'l', ' ', '\x95', '–', 'n', 'b', 'ò', '*', 'x', '´', 'ê', '.', 'ç', ',', 'ú', '5', '\uf0b7', 'î', ';', '¿', '¦', '3', 'æ', '[', 'ì', '%', ':', '¤', '_', 'u', '\t', 'ã', 'ō', '}', 'h', '®', 'g', 'å', 'r', 'ß', '\\', '<', '/', 'd', '\x85', 'ô', ')', '=', '»', 'i', '\x8d', '!', ']', '9', '\xad', 's', 'w', '@']
        self._pad_token_idx = 0
        self._unk_token_idx = 1
        self._cls_token = 2
        self._vocab_to_idx = {ch: i+3 for i, ch in enumerate(self._vocab)}

    @property
    def vocab_size(self) -> int:
        return len(self._vocab) + 3

    def encode(self, text: str, max_length: int = 512, **kwargs) -> List[int]:
        token_ids = [self._cls_token]

        for ch in text:
            token_ids.append(
                self._vocab_to_idx.get(ch, self._unk_token_idx)
            )

        token_ids = token_ids[:max_length]

        num_pads = max_length - len(token_ids)
        
        token_ids = token_ids + [self._pad_token_idx] * num_pads

        assert len(token_ids) == max_length

        return token_ids

if __name__ == '__main__':
    train_ds = load_dataset("sst2", split="train")
    #tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    #tokenizer = Tokenizer()
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    dbert = DistilBertModel.from_pretrained("distilbert-base-uncased")

    MAX_SEQ_LEN = 2**7

    def preprocess(data):
        data["token_ids"] = tokenizer.encode(
            data["sentence"],
            padding="max_length",
            add_special_tokens=True,
            truncation=True,
            max_length=MAX_SEQ_LEN,
        )
        data["features"] = dbert(torch.LongTensor(data["token_ids"]).unsqueeze(0)).last_hidden_state.squeeze(0)
        return data

    def collate_fn(batch_samples):
        features = []
        labels = []

        for data in batch_samples:
            features.append(data["features"])
            labels.append(data["label"])

        return {"features": torch.FloatTensor(features), "label": torch.LongTensor(labels)}

    train_ds = train_ds.shuffle().select(range(50)).map(preprocess)
    print(train_ds.shape)

    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=8)

    config = ModelConfig(
        num_features=128,
        num_heads=2,
        vocab_size=tokenizer.vocab_size,
        sliding_window_radius=16,
        segment_radius=1,
        hard_masking=True,
        global_token_ratio=16,
    )
    device = "cuda"

    model = VanillaTransformer(
        config.num_features,
        config.num_heads,
        768,
        config.sliding_window_radius,
        config.segment_radius,
        hard_masking=config.hard_masking,
        global_token_ratio=config.global_token_ratio,
        num_of_global_token_types=1,
        num_of_layers=2,
    )
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("number of parameters: ", num_params)

    model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

    for epoch in range(100):
        total_loss = []
        for batch in tqdm(train_dl):
            optimizer.zero_grad()

            logits = model.forward(batch["features"].to(device))

            loss = criterion(logits, batch["label"].to(device))

            loss.backward()
            optimizer.step()

            total_loss.append(loss.item())

        print("epoch {} loss -> {}".format(epoch+1, sum(total_loss) / len(total_loss)))