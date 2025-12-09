import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # Cần thêm thư viện này cho Beam Search
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
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
    MODEL_SAVE_PATH = "best_model.pt"
    TEST_EN_PATH = "Data/Test/test_2016_flickr.en"
    TEST_FR_PATH = "Data/Test/test_2016_flickr.fr"

    # Model Hyperparameters
    ENC_EMB_DIM = 512
    DEC_EMB_DIM = 512
    HID_DIM = 512
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    # Training Hyperparameters
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    N_EPOCHS = 15
    CLIP = 1
    PATIENCE = 3

    # Beam Search Hyperparameters
    BEAM_WIDTH = 3  # Độ rộng chùm (Số lượng nhánh giữ lại)

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
# 3. KIẾN TRÚC MÔ HÌNH (MODEL ARCHITECTURE)
# ==========================================

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):
        embedded = self.dropout(self.embedding(src))
        packed_embedded = pack_padded_sequence(embedded, src_len.cpu(), enforce_sorted=True)
        _, (hidden, cell) = self.rnn(packed_embedded)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, src_len, teacher_forcing_ratio=0.5):
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        hidden, cell = self.encoder(src, src_len)
        input_token = trg[0, :]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input_token, hidden, cell)
            outputs[t] = output
            top1 = output.argmax(1)
            input_token = trg[t] if random.random() < teacher_forcing_ratio else top1

        return outputs


enc = Encoder(len(vocab_en), Config.ENC_EMB_DIM, Config.HID_DIM, Config.N_LAYERS, Config.ENC_DROPOUT)
dec = Decoder(len(vocab_fr), Config.DEC_EMB_DIM, Config.HID_DIM, Config.N_LAYERS, Config.DEC_DROPOUT)
model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)


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


# --- BEAM SEARCH IMPLEMENTATION ---
def beam_search_decode(model, sentence, beam_width=3, max_len=50):
    model.eval()

    # 1. Encode
    tokens = [token for token in en_tokenizer(sentence)]
    indices = [vocab_en['<sos>']] + [vocab_en[t] for t in tokens] + [vocab_en['<eos>']]
    src_tensor = torch.LongTensor(indices).unsqueeze(1).to(DEVICE)
    src_len = torch.LongTensor([len(indices)]).to(DEVICE)

    with torch.no_grad():
        encoder_hidden, encoder_cell = model.encoder(src_tensor, src_len)

    # 2. Init Beam: [(score, sequence_indices, hidden, cell)]
    # Score dùng log probability nên bắt đầu bằng 0
    beam = [(0.0, [vocab_fr['<sos>']], encoder_hidden, encoder_cell)]

    # 3. Loop Decoding
    for _ in range(max_len):
        candidates = []

        # Duyệt qua tất cả các nhánh hiện có trong beam
        for score, seq, hidden, cell in beam:
            # Nếu nhánh đã kết thúc bằng <eos>, giữ nguyên
            if seq[-1] == vocab_fr['<eos>']:
                candidates.append((score, seq, hidden, cell))
                continue

            # Lấy token cuối cùng để làm input cho decoder
            input_token = torch.LongTensor([seq[-1]]).to(DEVICE)

            with torch.no_grad():
                output, new_hidden, new_cell = model.decoder(input_token, hidden, cell)

                # Tính log softmax để lấy xác suất log
                # output: [batch_size=1, output_dim]
                log_probs = F.log_softmax(output.squeeze(0), dim=0)

                # Lấy top k tokens có xác suất cao nhất
                topk_probs, topk_ids = torch.topk(log_probs, beam_width)

            # Tạo các nhánh con mới
            for i in range(beam_width):
                new_score = score + topk_probs[i].item()
                new_seq = seq + [topk_ids[i].item()]
                candidates.append((new_score, new_seq, new_hidden, new_cell))

        # Sắp xếp các candidates theo điểm số (cao nhất xếp trước)
        # Lưu ý: Vì là log prob nên giá trị là số âm, càng gần 0 càng lớn
        candidates.sort(key=lambda x: x[0], reverse=True)

        # Giữ lại 'beam_width' nhánh tốt nhất
        beam = candidates[:beam_width]

        # Nếu tất cả các nhánh trong beam đều đã kết thúc bằng <eos>, dừng sớm
        if all(seq[-1] == vocab_fr['<eos>'] for _, seq, _, _ in beam):
            break

    # 4. Chọn nhánh tốt nhất (đã được sort ở trên, lấy phần tử đầu tiên)
    # Có thể thêm logic chia cho độ dài câu (Length Normalization) để công bằng hơn
    best_seq = max(beam, key=lambda x: x[0] / len(x[1]))
    best_indices = best_seq[1]

    # Convert indices to string
    trg_tokens = [vocab_fr.lookup_token(i) for i in best_indices]

    # Remove <sos> and <eos>
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

        # SỬ DỤNG BEAM SEARCH
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

def draw_loss_chart(train_losses, val_losses, save_path="loss_chart_beamSearch.png"):
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
    print(f"\nBắt đầu huấn luyện {Config.N_EPOCHS} epochs (no Attention)...")
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