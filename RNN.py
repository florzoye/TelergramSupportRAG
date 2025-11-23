import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.nn import MultiheadAttention
from torch.nn.utils.rnn import pad_sequence

from sentence_transformers import SentenceTransformer


class PhraseDataset(data.Dataset):
    def __init__(self, path_true, path_false, transformer):
        self.transformer = transformer

        with open(path_true, 'r', encoding='utf-8') as f:
            phrase_true = [p.strip() for p in f.readlines()]

        with open(path_false, 'r', encoding='utf-8') as f:
            phrase_false = [p.strip() for p in f.readlines()]

        self.phrase_lst = [(p, 0) for p in phrase_true] + [(p, 1) for p in phrase_false]

    def __getitem__(self, idx):
        text, label = self.phrase_lst[idx]

        tokenizer = self.transformer.tokenizer
        transformer_module = self.transformer._first_module()

        encoded = tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=128
        )

        features = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"]
        }

        with torch.no_grad():
            out = transformer_module(features)

        # [seq_len, 384]
        token_embs = out["token_embeddings"].squeeze(0)

        emb = token_embs.to(torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        return emb, label

    def __len__(self):
        return len(self.phrase_lst)


def collate_fn(batch):
    sequences, labels = zip(*batch)

    sequences = [seq for seq in sequences]

    padded = pad_sequence(
        sequences,
        batch_first=True,
        padding_value=0.0
    )

    labels = torch.stack(labels)
    return padded, labels


class WordsRNN(nn.Module):
    def __init__(self, in_features, out_features, num_heads=4):
        super().__init__()
        self.hidden_size = 16
        
        self.mha = MultiheadAttention(
            embed_dim=self.hidden_size * 2,
            num_heads=num_heads,
            batch_first=True
        )

        self.rnn = nn.LSTM(
            in_features,
            self.hidden_size,
            batch_first=True,
            bidirectional=True
        )

        self.norm = nn.LayerNorm(self.hidden_size * 2)
        self.dropout = nn.Dropout(0.3)

        self.out = nn.Linear(self.hidden_size * 2, out_features)

    def forward(self, x):
        # [batch, seq_len, hidden*2]
        rnn_out, _ = self.rnn(x)
        rnn_out = self.dropout(rnn_out)

        attn_out, _ = self.mha(rnn_out, rnn_out, rnn_out)

        x = self.norm(rnn_out + attn_out)

        pooled = self.dropout(x.mean(dim=1))

        y = self.out(pooled)
        return y


transformer = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

dataset = PhraseDataset("data/train_data_true",
                        "data/train_data_false",
                        transformer)

train_loader = data.DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=collate_fn
)

model = WordsRNN(384, 1)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
loss_func = nn.BCEWithLogitsLoss()
epochs = 20


def train():
    model.train()
    
    for e in range(epochs):
        loss_mean = 0
        lm_count = 0

        loop = tqdm(train_loader)
        for x_train, y_train in loop:
            predict = model(x_train).squeeze(1)

            loss = loss_func(predict, y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lm_count += 1
            loss_mean = (1/lm_count) * loss.item() + (1 - 1/lm_count) * loss_mean
            loop.set_description(f"Epoch [{e+1}/{epochs}], loss={loss_mean:.3f}")

    torch.save(model.state_dict(), "data/model_rnn_bidir.tar")



def predict(phrase: str):
    model_path = "data/model_rnn_bidir.tar"

    if not os.path.exists(model_path):
        print("⚠ Модель не найдена. Запускаю обучение...")
        train()
        print("✅ Обучение завершено. Продолжаю предсказание.")

    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    tokenizer = transformer.tokenizer
    transformer_module = transformer._first_module()

    encoded = tokenizer(
        phrase,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=128
    )

    features = {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"]
    }

    with torch.no_grad():
        out = transformer_module(features)

    token_embs = out["token_embeddings"].squeeze(0)  # [seq_len, 384]

    x = token_embs.unsqueeze(0).to(torch.float32)

    with torch.no_grad():
        pred = model(x).squeeze(0)
        p = torch.sigmoid(pred)

    return p



# pred = predict('расскажи историю') 
# print(f'{pred.item()} = Относиться' if pred < 0.51 else f'{pred.item()} = нет')