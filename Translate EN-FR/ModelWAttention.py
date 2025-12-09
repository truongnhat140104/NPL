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
import math
import sys
import matplotlib.pyplot as plt

# ==========================================
# 1. CẤU HÌNH (CONFIGURATION)
# ==========================================
class Config:
    # File Paths
    TRAIN_EN_PATH = "Data/Train/train.en"
    TRAIN_FR_PATH = "Data/Train/train.fr"
    VAL_EN_PATH = "Data/Value/val.en"
    VAL_FR_PATH = "Data/Value/val.fr"
    MODEL_SAVE_PATH = "best_model_attention.pth"  # Đổi tên file save
    TEST_EN_PATH = "Data/Test/test_2016_flickr.en"
    TEST_FR_PATH = "Data/Test/test_2016_flickr.fr"

    # Model Hyperparameters
    ENC_EMB_DIM = 512 #256
    DEC_EMB_DIM = 512 # 256
    HID_DIM = 512
    N_LAYERS = 1  # Với Attention, thường dùng 1 layer LSTM để đơn giản hóa dimension
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    # Training Hyperparameters
    BATCH_SIZE = 64  # Giảm batch size vì Attention tốn VRAM hơn
    LEARNING_RATE = 0.001
    N_EPOCHS = 15
    CLIP = 1
    PATIENCE = 3

    # Beam Search
    BEAM_WIDTH = 3

    # Special Tokens
    SPECIAL_TOKENS = ['<unk>', '<pad>', '<sos>', '<eos>']


def get_device():
    if torch.cuda.is_available():
        print("🔧 Đang sử dụng thiết bị: NVIDIA CUDA (GPU)")
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        print("🔧 Đang sử dụng thiết bị: Apple Metal (MPS)")
        return torch.device('mps')
    else:
        print("🔧 Đang sử dụng thiết bị: CPU")
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

vocab_en = build_vocab_from_iterator(
    yield_tokens(train_data, en_tokenizer, 0),
    min_freq=1,
    specials=Config.SPECIAL_TOKENS,
    special_first=True
)
vocab_fr = build_vocab_from_iterator(
    yield_tokens(train_data, fr_tokenizer, 1),
    min_freq=1,
    specials=Config.SPECIAL_TOKENS,
    special_first=True
)

vocab_en.set_default_index(vocab_en['<unk>'])
vocab_fr.set_default_index(vocab_fr['<unk>'])

print(f"Vocab EN size: {len(vocab_en)}")
print(f"Vocab FR size: {len(vocab_fr)}")


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
    print("Không có dữ liệu Val riêng, sẽ cắt 10% từ tập Train.")
    train_size = int(0.9 * len(train_data))
    val_size = len(train_data) - train_size
    train_set, val_set = torch.utils.data.random_split(train_data, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=Config.BATCH_SIZE, collate_fn=collate_batch, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=Config.BATCH_SIZE, collate_fn=collate_batch)
else:
    val_loader = DataLoader(val_data, batch_size=Config.BATCH_SIZE, collate_fn=collate_batch)


# ==========================================
# 3. KIẾN TRÚC MÔ HÌNH VỚI ATTENTION (MODIFIED)
# ==========================================

# --- 1. ENCODER ---
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)

        # Bidirectional = True để lấy ngữ cảnh 2 chiều
        self.rnn = nn.LSTM(emb_dim, enc_hid_dim, bidirectional=True)

        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):
        # src: [src_len, batch_size]
        embedded = self.dropout(self.embedding(src))

        packed_embedded = pack_padded_sequence(embedded, src_len.cpu(), enforce_sorted=True)
        packed_outputs, (hidden, cell) = self.rnn(packed_embedded)
        outputs, _ = pad_packed_sequence(packed_outputs)

        # outputs: [src_len, batch, enc_hid_dim * 2] -> Dùng để tính Attention
        # hidden: [2, batch, enc_hid_dim] (Forward + Backward)

        # Gộp hidden state của 2 chiều forward/backward thành 1 để khớp với Decoder
        # hidden[-2,:,:] là Forward, hidden[-1,:,:] là Backward
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))

        # Chúng ta không dùng cell state của encoder cho decoder trong cấu trúc này,
        # hoặc có thể biến đổi tương tự hidden. Ở đây ta trả về hidden để init decoder.
        return outputs, hidden


# --- 2. ATTENTION ---
class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask):
        # hidden: [batch, dec_hid_dim]
        # encoder_outputs: [src_len, batch, enc_hid_dim * 2]

        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        # Lặp lại hidden src_len lần
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        # Permute encoder_outputs để khớp dimension
        encoder_outputs = encoder_outputs.permute(1, 0, 2)  # [batch, src_len, enc_hid_dim * 2]

        # Tính năng lượng (Energy)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        # Tính attention score
        attention = self.v(energy).squeeze(2)  # [batch, src_len]

        # Masking: Gán giá trị rất nhỏ vào các vị trí padding
        attention = attention.masked_fill(mask == 0, -1e10)

        return F.softmax(attention, dim=1)


# --- 3. DECODER ---
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
        # input: [batch]
        input = input.unsqueeze(0)  # [1, batch]
        embedded = self.dropout(self.embedding(input))

        # Tính Attention Weights
        a = self.attention(hidden, encoder_outputs, mask)  # [batch, src_len]
        a = a.unsqueeze(1)  # [batch, 1, src_len]

        encoder_outputs = encoder_outputs.permute(1, 0, 2)  # [batch, src_len, enc_hid * 2]

        # Tính Weighted sum (Context vector)
        weighted = torch.bmm(a, encoder_outputs)  # [batch, 1, enc_hid * 2]
        weighted = weighted.permute(1, 0, 2)  # [1, batch, enc_hid * 2]

        # Input cho RNN = Embedding + Context
        rnn_input = torch.cat((embedded, weighted), dim=2)

        output, (hidden, cell) = self.rnn(rnn_input, (hidden.unsqueeze(0), torch.zeros_like(hidden.unsqueeze(0))))
        # Lưu ý: LSTM Decoder layer=1, init cell=zeros hoặc học

        assert (output == hidden).all()

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))

        return prediction, hidden.squeeze(0)


# --- 4. SEQ2SEQ ---
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.device = device

    def create_mask(self, src):
        # src: [src_len, batch] -> mask: [batch, src_len]
        mask = (src != self.src_pad_idx).permute(1, 0)
        return mask

    def forward(self, src, trg, src_len, teacher_forcing_ratio=0.5):
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # Encoder outputs dùng cho attention
        encoder_outputs, hidden = self.encoder(src, src_len)

        # Tạo mask cho encoder outputs
        mask = self.create_mask(src)

        input_token = trg[0, :]

        for t in range(1, trg_len):
            output, hidden = self.decoder(input_token, hidden, encoder_outputs, mask)
            outputs[t] = output
            top1 = output.argmax(1)
            input_token = trg[t] if random.random() < teacher_forcing_ratio else top1

        return outputs


# --- KHỞI TẠO MÔ HÌNH ATTENTION ---
attn = Attention(Config.HID_DIM, Config.HID_DIM)
enc = Encoder(len(vocab_en), Config.ENC_EMB_DIM, Config.HID_DIM, Config.HID_DIM, Config.ENC_DROPOUT)
dec = Decoder(len(vocab_fr), Config.DEC_EMB_DIM, Config.HID_DIM, Config.HID_DIM, Config.DEC_DROPOUT, attn)

# Cần truyền src_pad_idx để làm Masking
model = Seq2Seq(enc, dec, vocab_en['<pad>'], DEVICE).to(DEVICE)


def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


model.apply(init_weights)

optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
criterion = nn.CrossEntropyLoss(ignore_index=vocab_fr['<pad>'])


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


# --- UPDATED BEAM SEARCH FOR ATTENTION ---
def beam_search_decode(model, sentence, beam_width=3, max_len=50):
    model.eval()

    # 1. Encode
    tokens = [token for token in en_tokenizer(sentence)]
    indices = [vocab_en['<sos>']] + [vocab_en[t] for t in tokens] + [vocab_en['<eos>']]
    src_tensor = torch.LongTensor(indices).unsqueeze(1).to(DEVICE)
    src_len = torch.LongTensor([len(indices)]).to(DEVICE)

    with torch.no_grad():
        # Lấy Encoder Outputs và Hidden đầu tiên
        encoder_outputs, hidden = model.encoder(src_tensor, src_len)

        # Tạo mask (vì batch size = 1 nên mask full true trừ khi sentence chỉ có pad)
        mask = model.create_mask(src_tensor)

    # 2. Init Beam: [(score, sequence_indices, hidden)]
    # Lưu ý: hidden của decoder trong model attention này shape là [batch, hid_dim] (do squeeze)
    beam = [(0.0, [vocab_fr['<sos>']], hidden)]

    # 3. Loop Decoding
    for _ in range(max_len):
        candidates = []

        for score, seq, curr_hidden in beam:
            if seq[-1] == vocab_fr['<eos>']:
                candidates.append((score, seq, curr_hidden))
                continue

            input_token = torch.LongTensor([seq[-1]]).to(DEVICE)

            with torch.no_grad():
                # Truyền encoder_outputs và mask vào decoder
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


def calculate_bleu_on_test_set(model, test_en_path, test_fr_path):
    print(f"\n--- BẮT ĐẦU ĐÁNH GIÁ (Beam Width={Config.BEAM_WIDTH}) ---")
    model.eval()

    with open(test_en_path, 'r', encoding='utf-8') as f:
        test_en = [line.strip() for line in f]
    with open(test_fr_path, 'r', encoding='utf-8') as f:
        test_fr = [line.strip() for line in f]

    predictions = []
    references = []

    for i in range(len(test_en)):
        src = test_en[i]
        trg = test_fr[i]

        pred_sent = beam_search_decode(model, src, beam_width=Config.BEAM_WIDTH)

        predictions.append(fr_tokenizer(pred_sent))
        references.append([fr_tokenizer(trg)])

        if (i + 1) % 50 == 0:
            print(f"Đã xử lý {i + 1}/{len(test_en)} câu...")

    score = corpus_bleu(references, predictions)
    print(f"------------------------------------------------")
    print(f"TEST SET BLEU SCORE: {score * 100:.2f}")
    print(f"------------------------------------------------")


def translate_custom_sentences(model, sentence_pairs):
    print(f"\n{'=' * 20} DỊCH 5 CÂU TỰ CHỌN (KÈM ĐÁP ÁN) {'=' * 20}")
    model.eval()

    for i, (src, ref) in enumerate(sentence_pairs):
        start_time = time.time()

        # Dịch
        pred = beam_search_decode(model, src, beam_width=Config.BEAM_WIDTH)

        end_time = time.time()

        print(f"Custom #{i + 1} (Time: {end_time - start_time:.2f}s)")
        print(f" Input : {src}")
        print(f" Ref   : {ref}")  # Đáp án chuẩn
        print(f" Pred  : {pred}")  # Máy dịch

        # So sánh nhanh xem đúng không
        if ref.lower().strip() == pred.lower().strip():
            print("  Evaluation: PERFECT!")
        else:
            print("  Evaluation: Different")

        print("-" * 60)


def draw_loss_chart(train_losses, val_losses, save_path="loss_chart.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', marker='o', color='blue')
    plt.plot(val_losses, label='Validation Loss', marker='o', color='red')

    plt.title('Training & Validation Loss History')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path)
    plt.close()  # Đóng plot để giải phóng bộ nhớ
    print(f"\n📊 Đã lưu biểu đồ loss tại: {save_path}")

# ==========================================
# 5. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    print(f"\nBắt đầu huấn luyện {Config.N_EPOCHS} epochs (với Attention)...")
    best_valid_loss = float('inf')
    no_improve_epoch = 0

    train_history = []
    valid_history = []

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
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            no_improve_epoch = 0
            print(f'Epoch: {epoch + 1:02} | Time: {int(mins)}m {int(secs)}s | ✅ Save Best Model')
        else:
            no_improve_epoch += 1
            print(
                f'Epoch: {epoch + 1:02} | Time: {int(mins)}m {int(secs)}s | ⚠️ No improve ({no_improve_epoch}/{Config.PATIENCE})')

        print(f'\tTrain Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f}')

        if no_improve_epoch >= Config.PATIENCE:
            print("🛑 Early Stopping!")
            break

    draw_loss_chart(train_history, valid_history)

    print("\nĐang load lại model tốt nhất để đánh giá...")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=DEVICE))

    calculate_bleu_on_test_set(model, Config.TEST_EN_PATH, Config.TEST_FR_PATH)

    my_sentences = [
        ("A black dog is running on the grass.", "Un chien noir court sur l'herbe."),
        ("Two men are playing soccer in the park.", "Deux hommes jouent au football dans le parc."),
        ("The woman in a red dress is reading a book.", "La femme à la robe rouge lit un livre."),
        ("A little girl is eating an apple.", "Une petite fille mange une pomme."),
        ("People are walking down the street.", "Les gens marchent dans la rue.")
    ]

    translate_custom_sentences(model, my_sentences)