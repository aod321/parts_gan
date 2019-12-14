import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim

from model import Generator, Parts_Discriminator
import torch
import argparse
import uuid
import numpy as np
import torchvision
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from dataset import FaceImageDataset
from torchvision import transforms
from preprogress import Resize, ToTensor
from torch.utils.data import DataLoader

uuid_8 = str(uuid.uuid1())[0:9]
parser = argparse.ArgumentParser()
parser.add_argument("--cuda", default=0, type=int, help="Which GPU to train.")
parser.add_argument("--batch_size", default=16, type=int, help="Batch size to use during training.")
parser.add_argument("--display_freq", default=10, type=int, help="Display frequency")
parser.add_argument("--lr", default=0.02, type=float, help="Learning rate for optimizer")
parser.add_argument("--epochs", default=250, type=int, help="Number of epochs to train")
parser.add_argument("--eval_per_epoch", default=1, type=int, help="eval_per_epoch ")
parser.add_argument("--workers", default=16, type=int, help="Workers")
parser.add_argument("--b1", default=0.9, type=float, help="Adam b1")
parser.add_argument("--b2", default=0.999, type=float, help="Adam b2")
parser.add_argument("--sample_interval", type=int, default=50, help="interval between image sampling")
args = parser.parse_args()
print(args)
writer = SummaryWriter('runs')
device = torch.device("cuda:%d" % args.cuda if torch.cuda.is_available() else "cpu")
generator = Generator().to(device)
discriminator = Parts_Discriminator().to(device)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

adversarial_loss = nn.BCEWithLogitsLoss()
classify_loss = nn.CrossEntropyLoss()

txt_file = "exemplars.txt"
# Dataset Read_in Part
img_root_dir = "/data1/yinzi/datas"
# root_dir = '/home/yinzi/Downloads/datas'
part_root_dir = "/data1/yinzi/facial_parts"
dataset = FaceImageDataset(txt_file=txt_file,
                           img_root_dir=img_root_dir,
                           part_root_dir=part_root_dir,
                           transform=transforms.Compose([
                               Resize((64, 64)),
                               ToTensor()
                           ])
                           )
dataloader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=args.workers)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


# helper function to show an image
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


# ----------
#  Training
# ----------
def train():
    for epoch in range(args.epochs):
        for i, batch in enumerate(dataloader):
            imgs = batch['image'].to(device)
            real_parts = batch['parts'].to(device)
            batch_size = imgs.shape[0]
            # Adversarial ground truths
            valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
            fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)
            #
            # # Configure input
            # real_imgs = Variable(imgs.type(torch.FloatTensor))
            # labels = Variable(labels.type(torch.LongTensor))
            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            # Generate a batch of images
            gen_parts = generator(imgs)

            real_labels = {
                'eye1': Variable(LongTensor(batch_size).fill_(0), requires_grad=False),
                'eye2': Variable(LongTensor(batch_size).fill_(1), requires_grad=False),
                'nose': Variable(LongTensor(batch_size).fill_(2), requires_grad=False),
                'mouth': Variable(LongTensor(batch_size).fill_(3), requires_grad=False)
            }

            # Loss measures generator's ability to fool the discriminator
            eye1, eye2, nose, mouth = discriminator(gen_parts)
            eye1_g_loss = adversarial_loss(eye1[0], valid)
            eye2_g_loss = adversarial_loss(eye2[0], valid)
            nose_g_loss = adversarial_loss(nose[0], valid)
            mouth_g_loss = adversarial_loss(mouth[0], valid)

            # Classify Loss
            # c_eye1_loss = classify_loss(eye1[1], real_labels['eye1'])
            # c_eye2_loss = classify_loss(eye2[1], real_labels['eye2'])
            # c_nose_loss = classify_loss(nose[1], real_labels['nose'])
            # c_mouth_loss = classify_loss(mouth[1], real_labels['mouth'])
            # c_g_loss = (c_eye1_loss + c_eye2_loss + c_mouth_loss + c_nose_loss) / 4.0
            g_loss = (eye1_g_loss + eye2_g_loss + nose_g_loss + mouth_g_loss) / 4.0
            # loss = (g_loss + c_g_loss) / 2.0
            loss = g_loss
            loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Loss for real images
            real_eye1_pred, real_eye2_pred, real_nose_pred, real_mouth_pred = discriminator(real_parts)
            d_eye1_real_loss = adversarial_loss(real_eye1_pred[0], valid)
            d_eye2_real_loss = adversarial_loss(real_eye2_pred[0], valid)
            d_nose_real_loss = adversarial_loss(real_nose_pred[0], valid)
            d_mouth_real_loss = adversarial_loss(real_mouth_pred[0], valid)
            d_real_loss = (d_eye1_real_loss + d_eye2_real_loss + d_nose_real_loss + d_mouth_real_loss) / 4.0
            # Classify Loss
            c_eye1_loss = classify_loss(real_eye1_pred[1], real_labels['eye1'])
            c_eye2_loss = classify_loss(real_eye2_pred[1], real_labels['eye2'])
            c_nose_loss = classify_loss(real_nose_pred[1], real_labels['nose'])
            c_mouth_loss = classify_loss(real_mouth_pred[1], real_labels['mouth'])
            c_real_loss = (c_eye1_loss + c_eye2_loss + c_mouth_loss + c_nose_loss) / 4.0

            # Loss for fake images
            fake_eye1_pred, fake_eye2_pred, fake_nose_pred, fake_mouth_pred = discriminator(gen_parts.detach())
            d_fake_eye1_loss = adversarial_loss(fake_eye1_pred[0], fake)
            d_fake_eye2_loss = adversarial_loss(fake_eye2_pred[0], fake)
            d_fake_nose_loss = adversarial_loss(fake_nose_pred[0], fake)
            d_fake_mouth_loss = adversarial_loss(fake_mouth_pred[0], fake)
            d_fake_loss = (d_fake_eye1_loss + d_fake_eye2_loss + d_fake_nose_loss + d_fake_mouth_loss) / 4.0

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss + c_real_loss) / 3.0

            # Calculate discriminator accuracy
            d_acc = []
            eye1_pred = fake_eye1_pred[1].detach()
            eye1_gt = real_labels['eye1'].detach()
            d_acc.append(torch.mean((torch.argmax(eye1_pred, dim=1) == eye1_gt).float()).tolist())
            eye2_pred = fake_eye2_pred[1].detach()
            eye2_gt = real_labels['eye2'].detach()
            d_acc.append(torch.mean((torch.argmax(eye2_pred, dim=1) == eye2_gt).float()).tolist())
            nose_pred = fake_nose_pred[1].detach()
            nose_gt = real_labels['nose'].detach()
            d_acc.append(torch.mean((torch.argmax(nose_pred, dim=1) == nose_gt).float()).tolist())
            mouth_pred = fake_mouth_pred[1].detach()
            mouth_gt = real_labels['mouth'].detach()
            d_acc.append(torch.mean((torch.argmax(mouth_pred, dim=1) == mouth_gt).float()).tolist())
            d_loss.backward()
            optimizer_D.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f][G loss: %f]\n"
                "[acc_eye1: %d%%,acc_eye2: %d%%,acc_nose: %d%%,acc_mouth: %d%%]"
                % (epoch, args.epochs, i, len(dataloader), d_loss.item(), g_loss.item(),
                   100 * d_acc[0], 100 * d_acc[1], 100 * d_acc[2], 100 * d_acc[3])
            )
            batches_done = epoch * len(dataloader) + i
            if batches_done % args.sample_interval == 0:
                print("show image")
                for p in range(4):
                    img_grid = torchvision.utils.make_grid(gen_parts[:, p])
                    # matplotlib_imshow(img_grid, one_channel=False)
                    writer.add_image('Generated Image%d' % p, img_grid)

    # Save Model
    def save_state(model, fname):
        if isinstance(model, torch.nn.DataParallel):
            state = model.module.state_dict()
        else:
            state = model.state_dict()

        torch.save(state, fname)
        print('save model at {}'.format(fname))

    save_state(generator, "Gen.pth.tar")
    save_state(discriminator, "disc.pth.tar")
    print("All Done!")

train()