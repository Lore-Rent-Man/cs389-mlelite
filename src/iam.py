import sys

import Levenshtein as leven
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from colorama import Fore
from skimage.color import rgb2gray
from skimage.transform import rotate
from torch import nn, optim
from torch.utils.data import SubsetRandomSampler, DataLoader
from tqdm import tqdm

from dataset import IAMData
from model import IAMModel

np.random.seed(42)
torch.manual_seed(42)
dev = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================= PREPARING DATASET ======================================================
dataset = IAMData(txt_file='../dataset/lines.txt',
                  root_dir='../dataset',
                  output_size=(128, 1200),
                  border_pad=(4, 10))

classes = ''.join(dataset.char_dict.keys())
text_file = open("chars.txt", "w", encoding='utf-8')
text_file.write('\n'.join([x if x != '#' else '\\#' for x in dataset.char_dict.keys()]))
text_file.close()


def collate(batch):
    images, words = [b.get('image') for b in batch], [b.get('word') for b in batch]
    images = torch.stack(images, 0)
    # Calculate target lengths for the current batch
    lengths = [len(word) for word in words]
    # According to https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html
    # Tensor of size sum(target_lengths) the targets are assumed to be un-padded and concatenated within 1 dimension.
    targets = torch.empty(sum(lengths)).fill_(len(classes)).long()
    lengths = torch.tensor(lengths)
    # Now we need to fill targets according to calculated lengths
    for j, word in enumerate(words):
        start = sum(lengths[:j])
        end = lengths[j]
        targets[start:start + end] = torch.tensor([dataset.char_dict.get(letter) for letter in word]).long()
    return images.to(dev), targets.to(dev), lengths.to(dev)


# ================================================= MODEL ==============================================================
model = IAMModel(time_step=96,
                 feature_size=512,
                 hidden_size=512,
                 output_size=len(classes) + 1,
                 num_rnn_layers=4)
model.to(dev)


# ================================================ TRAINING MODEL ======================================================
def fit(model, update_interval, epochs, train_data_loader, valid_data_loader, lr=1e-3, wd=1e-2, betas=(0.9, 0.999)):
    best_leven = 1000
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
                     weight_decay=wd, betas=betas)
    opt.zero_grad(set_to_none=False)
    len_train = len(train_data_loader)
    loss_func = nn.CTCLoss(reduction='sum', zero_infinity=True, blank=len(classes))
    losses = []
    for i in range(1, epochs + 1):
        # ============================================ TRAINING ========================================================
        batch_n = 1
        train_levenshtein = 0
        len_levenshtein = 0
        counter = 0
        for xb, yb, lens in tqdm(train_data_loader,
                                 position=0, leave=True,
                                 file=sys.stdout, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET)):
            model.train()
            # And the lengths are specified for each sequence to achieve masking
            # under the assumption that sequences are padded to equal lengths.
            input_lengths = torch.full((xb.size()[0],), model.time_step, dtype=torch.long)
            loss = loss_func(model(xb).log_softmax(2).requires_grad_(), yb, input_lengths, lens)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=False)
            if counter % update_interval == 0 and loss:
                losses.append(round(loss.item(), 2))
            
            # ================================== TRAINING LEVENSHTEIN DISTANCE =========================================
            if batch_n > (len_train - 5):
                model.eval()
                with torch.no_grad():
                    decoded = model.beam_search_with_lm(xb)
                    for j in range(0, len(decoded)):
                        # We need to find actual string somewhere in the middle of the 'targets'
                        # tensor having tensor 'lens' with known lengths
                        actual = yb.cpu().numpy()[0 + sum(lens[:j]): sum(lens[:j]) + lens[j]]
                        train_levenshtein += leven.distance(''.join([letter for letter in decoded[j]]), ''.join([decode_map.get(letter.item()) for letter in actual[:]]))
                    len_levenshtein += sum(lens).item()
            batch_n += 1
            counter += 1

        # ============================================ VALIDATION ======================================================
        model.eval()
        with torch.no_grad():
            val_levenshtein = 0
            target_lengths = 0
            for xb, yb, lens in tqdm(valid_data_loader,
                                     position=0, leave=True,
                                     file=sys.stdout, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET)):
                decoded = model.beam_search_with_lm(xb)
                for j in range(0, len(decoded)):
                    actual = yb.cpu().numpy()[0 + sum(lens[:j]): sum(lens[:j]) + lens[j]]
                    val_levenshtein += leven.distance(''.join([letter for letter in decoded[j]]), ''.join([decode_map.get(letter.item()) for letter in actual[:]]))
                target_lengths += sum(lens).item()

        print('epoch {}: Train Levenshtein {} | Validation Levenshtein {}'
              .format(i, train_levenshtein / len_levenshtein, val_levenshtein / target_lengths), end='\n')
        # ============================================ SAVE MODEL ======================================================
        if (val_levenshtein / target_lengths) < best_leven:
            torch.save(model.state_dict(), f=str((val_levenshtein / target_lengths) * 100).replace('.', '_') + '_' + 'model.pth')
            best_leven = val_levenshtein / target_lengths
    return losses

train_batch_size = 30
update_interval = 5
validation_batch_size = 40
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(0.2 * dataset_size))
decode_map = {v: k for k, v in dataset.char_dict.items()}
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(dataset, batch_size=train_batch_size, sampler=train_sampler, collate_fn=collate)
validation_loader = DataLoader(dataset, batch_size=validation_batch_size, sampler=valid_sampler, collate_fn=collate)
print("Training...")
losses = fit(model=model, update_interval=update_interval, epochs=20, train_data_loader=train_loader, valid_data_loader=validation_loader)
print(losses)
plt.plot(np.arange(len(losses)) * train_batch_size * update_interval, losses)
plt.title("training curve for english and chinese dataset")
plt.xlabel("number of images trained on")
plt.ylabel("CTC loss")
plt.show()

# ============================================ TESTING =================================================================
def batch_predict(model, valid_dl, up_to):
    xb, yb, lens = iter(valid_dl).next()
    model.eval()
    with torch.no_grad():
        outs = model.beam_search_with_lm(xb)
        for i in range(len(outs)):
            start = sum(lens[:i])
            end = lens[i].item()
            corr = ''.join([decode_map.get(letter.item()) for letter in yb[start:start + end]])
            predicted = ''.join([letter for letter in outs[i]])
            # ============================================ SHOW IMAGE ==================================================
            img = xb[i, :, :, :].permute(1, 2, 0).cpu().numpy()
            img = rgb2gray(img)
            img = rotate(img, angle=90, clip=False, resize=True)
            f, ax = plt.subplots(1, 1)
            mpl.rcParams["font.size"] = 8
            ax.imshow(img, cmap='gray')
            mpl.rcParams["font.size"] = 14
            plt.gcf().text(x=0.1, y=0.1, s="Actual: " + str(corr))
            plt.gcf().text(x=0.1, y=0.2, s="Predicted: " + str(predicted))
            f.set_size_inches(10, 3)
            print('actual: {}'.format(corr))
            print('predicted:   {}'.format(predicted))
            if i + 1 == up_to:
                break
    plt.show()


batch_predict(model=model, valid_dl=validation_loader, up_to=20)
