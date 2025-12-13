import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from nltk.translate.bleu_score import corpus_bleu
import random
import time
import sys
import matplotlib.pyplot as plt
import os
import numpy as np  # Thêm numpy để tính trung bình


# ==========================================
# 1. CẤU HÌNH (CONFIGURATION)
# ==========================================
class Config:
    # File Paths
    TRAIN_EN_PATH = "Data/Train/train.en"
    TRAIN_FR_PATH = "Data/Train/train.fr"
    VAL_EN_PATH = "Data/Value/val.en"
    VAL_FR_PATH = "Data/Value/val.fr"

    # Folder lưu kết quả
    GRAPH_SAVE_DIR = "Graph/Attention"

    TEST_EN_PATH = "Data/Test/test_2016_flickr.en"
    TEST_FR_PATH = "Data/Test/test_2016_flickr.fr"

    # Model Hyperparameters
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 256
    N_LAYERS = 1
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    # Training Hyperparameters
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    N_EPOCHS = 15
    CLIP = 1
    PATIENCE = 3

    NUM_RUNS = 10

    # Beam Search
    BEAM_WIDTH = 3

    # Special Tokens
    SPECIAL_TOKENS = ['<unk>', '<pad>', '<sos>', '<eos>']


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


DEVICE = get_device()

# ==========================================
# 2. XỬ LÝ DỮ LIỆU (DATA PROCESSING)
# ==========================================
print("\n--- Đang xử lý dữ liệu ---")

try:
    en_tokenizer = get_tokenizer("spacy", language="en_core_web_sm")
    fr_tokenizer = get_tokenizer("spacy", language="fr_core_news_sm")
except OSError:
    print("Vui lòng cài đặt spacy models.")
    sys.exit()


def read_data(path_en, path_fr):
    with open(path_en, "r", encoding="utf-8") as f:
        data_en = [line.strip() for line in f]
    with open(path_fr, "r", encoding="utf-8") as f:
        data_fr = [line.strip() for line in f]
    return list(zip(data_en, data_fr))


def yield_tokens(data_iterator, tokenizer, index):
    for data_sample in data_iterator:
        yield tokenizer(data_sample[index])


train_data = read_data(Config.TRAIN_EN_PATH, Config.TRAIN_FR_PATH)
val_data = read_data(Config.VAL_EN_PATH, Config.VAL_FR_PATH)
test_data = read_data(Config.TEST_EN_PATH, Config.TEST_FR_PATH)

vocab_en = build_vocab_from_iterator(yield_tokens(train_data, en_tokenizer, 0), min_freq=1,
                                     specials=Config.SPECIAL_TOKENS, special_first=True)
vocab_fr = build_vocab_from_iterator(yield_tokens(train_data, fr_tokenizer, 1), min_freq=1,
                                     specials=Config.SPECIAL_TOKENS, special_first=True)

vocab_en.set_default_index(vocab_en['<unk>'])
vocab_fr.set_default_index(vocab_fr['<unk>'])


def text_pipeline(text, tokenizer, vocab):
    tokens = tokenizer(text)
    indices = [vocab['<sos>']] + vocab(tokens) + [vocab['<eos>']]
    return torch.tensor(indices, dtype=torch.long)


def collate_batch(batch):
    processed_batch = []
    for src_text, trg_text in batch:
        src_tensor = text_pipeline(src_text, en_tokenizer, vocab_en)
        trg_tensor = text_pipeline(trg_text, fr_tokenizer, vocab_fr)
        processed_batch.append((src_tensor, trg_tensor, len(src_tensor)))
    processed_batch.sort(key=lambda x: x[2], reverse=True)
    src_list, trg_list, src_lens = zip(*processed_batch)
    src_batch = pad_sequence(src_list, padding_value=vocab_en['<pad>'])
    trg_batch = pad_sequence(trg_list, padding_value=vocab_fr['<pad>'])
    src_lens = torch.tensor(src_lens, dtype=torch.int64)
    return src_batch, trg_batch, src_lens


train_loader = DataLoader(train_data, batch_size=Config.BATCH_SIZE, collate_fn=collate_batch, shuffle=True)
if len(val_data) == 0:
    train_size = int(0.9 * len(train_data))
    val_size = len(train_data) - train_size
    train_set, val_set = torch.utils.data.random_split(train_data, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=Config.BATCH_SIZE, collate_fn=collate_batch, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=Config.BATCH_SIZE, collate_fn=collate_batch)
else:
    val_loader = DataLoader(val_data, batch_size=Config.BATCH_SIZE, collate_fn=collate_batch)


# ==========================================
# 3. KIẾN TRÚC MÔ HÌNH VỚI ATTENTION
# ==========================================
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, enc_hid_dim, bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):
        embedded = self.dropout(self.embedding(src))
        packed_embedded = pack_padded_sequence(embedded, src_len.cpu(), enforce_sorted=True)
        packed_outputs, (hidden, cell) = self.rnn(packed_embedded)
        outputs, _ = pad_packed_sequence(packed_outputs)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask):
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        attention = attention.masked_fill(mask == 0, -1e10)
        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs, mask):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        a = self.attention(hidden, encoder_outputs, mask)
        a = a.unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted = torch.bmm(a, encoder_outputs)
        weighted = weighted.permute(1, 0, 2)
        rnn_input = torch.cat((embedded, weighted), dim=2)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden.unsqueeze(0), torch.zeros_like(hidden.unsqueeze(0))))
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))
        return prediction, hidden.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.device = device

    def create_mask(self, src):
        mask = (src != self.src_pad_idx).permute(1, 0)
        return mask

    def forward(self, src, trg, src_len, teacher_forcing_ratio=0.5):
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src, src_len)
        mask = self.create_mask(src)
        input_token = trg[0, :]
        for t in range(1, trg_len):
            output, hidden = self.decoder(input_token, hidden, encoder_outputs, mask)
            outputs[t] = output
            top1 = output.argmax(1)
            input_token = trg[t] if random.random() < teacher_forcing_ratio else top1
        return outputs


def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


# ==========================================
# 4. TRAINING & BEAM SEARCH UTILITIES
# ==========================================
def train_epoch(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for src, trg, src_len in iterator:
        src, trg, src_len = src.to(DEVICE), trg.to(DEVICE), src_len.to(DEVICE)
        optimizer.zero_grad()
        output = model(src, trg, src_len)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def evaluate_epoch(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for src, trg, src_len in iterator:
            src, trg, src_len = src.to(DEVICE), trg.to(DEVICE), src_len.to(DEVICE)
            output = model(src, trg, src_len, teacher_forcing_ratio=0)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def beam_search_decode(model, sentence, beam_width=3, max_len=50):
    model.eval()
    tokens = [token for token in en_tokenizer(sentence)]
    indices = [vocab_en['<sos>']] + [vocab_en[t] for t in tokens] + [vocab_en['<eos>']]
    src_tensor = torch.LongTensor(indices).unsqueeze(1).to(DEVICE)
    src_len = torch.LongTensor([len(indices)]).to(DEVICE)
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor, src_len)
        mask = model.create_mask(src_tensor)
    beam = [(0.0, [vocab_fr['<sos>']], hidden)]
    for _ in range(max_len):
        candidates = []
        for score, seq, curr_hidden in beam:
            if seq[-1] == vocab_fr['<eos>']:
                candidates.append((score, seq, curr_hidden))
                continue
            input_token = torch.LongTensor([seq[-1]]).to(DEVICE)
            with torch.no_grad():
                output, new_hidden = model.decoder(input_token, curr_hidden, encoder_outputs, mask)
                log_probs = F.log_softmax(output.squeeze(0), dim=0)
                topk_probs, topk_ids = torch.topk(log_probs, beam_width)
            for i in range(beam_width):
                new_score = score + topk_probs[i].item()
                new_seq = seq + [topk_ids[i].item()]
                candidates.append((new_score, new_seq, new_hidden))
        candidates.sort(key=lambda x: x[0], reverse=True)
        beam = candidates[:beam_width]
        if all(seq[-1] == vocab_fr['<eos>'] for _, seq, _ in beam):
            break
    best_seq = max(beam, key=lambda x: x[0] / len(x[1]))
    best_indices = best_seq[1]
    trg_tokens = [vocab_fr.lookup_token(i) for i in best_indices]
    if trg_tokens[0] == '<sos>': trg_tokens.pop(0)
    if trg_tokens[-1] == '<eos>': trg_tokens.pop(-1)
    return " ".join(trg_tokens)


def calculate_bleu_on_test_set(model, test_en_path, test_fr_path, beam_width=3):
    print(f"   [Evaluating BLEU with Beam={beam_width} - This may take a while...]")
    model.eval()
    with open(test_en_path, 'r', encoding='utf-8') as f:
        test_en = [line.strip() for line in f]
    with open(test_fr_path, 'r', encoding='utf-8') as f:
        test_fr = [line.strip() for line in f]
    predictions = []
    references = []
    # Để tiết kiệm thời gian, có thể chỉ test 100 câu đầu nếu cần nhanh: range(min(100, len(test_en)))
    for i in range(len(test_en)):
        src = test_en[i]
        trg = test_fr[i]
        pred_sent = beam_search_decode(model, src, beam_width=beam_width)
        predictions.append(fr_tokenizer(pred_sent))
        references.append([fr_tokenizer(trg)])
    score = corpus_bleu(references, predictions)
    return score


def translate_custom_sentences(model, sentence_pairs):
    print(f"\n   --- Custom Translations ---")
    model.eval()
    for i, (src, ref) in enumerate(sentence_pairs):
        pred = beam_search_decode(model, src, beam_width=Config.BEAM_WIDTH)
        print(f"   In: {src}")
        print(f"   Out: {pred}")


def draw_loss_chart(train_losses, val_losses, run_id, save_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', marker='o', color='blue')
    plt.plot(val_losses, label='Validation Loss', marker='o', color='red')
    plt.title(f'Training & Validation Loss - RUN {run_id}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    file_name = f"loss_chart_run_{run_id}.png"
    save_path = os.path.join(save_dir, file_name)
    plt.savefig(save_path)
    plt.close()
    print(f"   📊 Đã lưu biểu đồ tại: {save_path}")


# ==========================================
# 5. MAIN EXECUTION (VÒNG LẶP 10 LẦN)
# ==========================================

if __name__ == "__main__":
    if not os.path.exists(Config.GRAPH_SAVE_DIR):
        os.makedirs(Config.GRAPH_SAVE_DIR)
        print(f"Đã tạo thư mục: {Config.GRAPH_SAVE_DIR}")

    print(f"\n🚀 BẮT ĐẦU CHẠY THỰC NGHIỆM {Config.NUM_RUNS} LẦN ĐỘC LẬP")

    # List để lưu kết quả BLEU của từng Run
    all_runs_bleu = []

    # Danh sách câu test nhanh
    my_sentences = [
        ("A black dog is running on the grass.", "Un chien noir court sur l'herbe."),
        ("Two men are playing soccer in the park.", "Deux hommes jouent au football dans le parc."),
        ("The woman in a red dress is reading a book.", "La femme à la robe rouge lit un livre."),
        ("A little girl is eating an apple.", "Une petite fille mange une pomme."),
        ("People are walking down the street.", "Les gens marchent dans la rue.")
    ]

    for run_i in range(1, Config.NUM_RUNS + 1):
        print(f"\n{'=' * 20} RUN {run_i}/{Config.NUM_RUNS} {'=' * 20}")

        # --- KHỞI TẠO MỚI ---
        attn = Attention(Config.HID_DIM, Config.HID_DIM)
        enc = Encoder(len(vocab_en), Config.ENC_EMB_DIM, Config.HID_DIM, Config.HID_DIM, Config.ENC_DROPOUT)
        dec = Decoder(len(vocab_fr), Config.DEC_EMB_DIM, Config.HID_DIM, Config.HID_DIM, Config.DEC_DROPOUT, attn)
        model = Seq2Seq(enc, dec, vocab_en['<pad>'], DEVICE).to(DEVICE)
        model.apply(init_weights)

        optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
        criterion = nn.CrossEntropyLoss(ignore_index=vocab_fr['<pad>'])

        current_model_path = f"best_model_attention_run_{run_i}.pth"

        best_valid_loss = float('inf')
        no_improve_epoch = 0
        train_history = []
        valid_history = []

        # --- TRAINING LOOP ---
        for epoch in range(Config.N_EPOCHS):
            start_time = time.time()
            train_loss = train_epoch(model, train_loader, optimizer, criterion, Config.CLIP)
            valid_loss = evaluate_epoch(model, val_loader, criterion)
            train_history.append(train_loss)
            valid_history.append(valid_loss)
            scheduler.step(valid_loss)

            end_time = time.time()
            mins, secs = divmod(end_time - start_time, 60)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), current_model_path)
                no_improve_epoch = 0
                save_msg = "✅ Save Best"
            else:
                no_improve_epoch += 1
                save_msg = f"⚠️ No improve ({no_improve_epoch}/{Config.PATIENCE})"

            print(
                f'   Ep {epoch + 1:02} | {int(mins)}m {int(secs)}s | Tr: {train_loss:.3f} | Val: {valid_loss:.3f} | {save_msg}')

            if no_improve_epoch >= Config.PATIENCE:
                print("   🛑 Early Stopping!")
                break

        draw_loss_chart(train_history, valid_history, run_i, Config.GRAPH_SAVE_DIR)

        # --- ĐÁNH GIÁ (EVALUATION) CHO RUN NÀY ---
        print(f"\n Đang đánh giá Run {run_i}...")
        model.load_state_dict(torch.load(current_model_path, map_location=DEVICE))

        # 1. Tính BLEU trên tập test
        bleu_score = calculate_bleu_on_test_set(model, Config.TEST_EN_PATH, Config.TEST_FR_PATH, Config.BEAM_WIDTH)
        all_runs_bleu.append(bleu_score)
        print(f" BLEU Score (Run {run_i}): {bleu_score * 100:.2f}")

        # 2. Dịch thử vài câu
        translate_custom_sentences(model, my_sentences)

        # Dọn dẹp bộ nhớ GPU
        del model, optimizer, scheduler, criterion
        torch.cuda.empty_cache()

    # --- TỔNG KẾT ---
    print(f"\n{'=' * 40}")
    print(f"KẾT QUẢ TỔNG HỢP SAU {Config.NUM_RUNS} LẦN CHẠY")
    print(f"{'=' * 40}")
    print(f"Chi tiết BLEU từng run: {[round(b * 100, 2) for b in all_runs_bleu]}")
    print(f"Trung bình cộng (Mean BLEU): {np.mean(all_runs_bleu) * 100:.2f}")
    print(f"Độ lệch chuẩn (Std Dev): {np.std(all_runs_bleu) * 100:.2f}")
    print(f"Cao nhất (Max): {np.max(all_runs_bleu) * 100:.2f}")
    print(f"Thấp nhất (Min): {np.min(all_runs_bleu) * 100:.2f}")