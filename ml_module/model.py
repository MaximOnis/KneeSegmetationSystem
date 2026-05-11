import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from data_module.DataLoader import load_dataset


# U-NET
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Decoder

        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)

        self.dec4 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)

        self.dec3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)

        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)

        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.conv_last = nn.Conv2d(32, 5, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        b = self.bottleneck(self.pool4(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.conv_last(d1)


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, outputs, targets):

        num_classes = outputs.shape[1]

        # softmax probabilities
        outputs = F.softmax(outputs, dim=1)

        # one-hot targets
        targets_onehot = F.one_hot(
            targets,
            num_classes=num_classes
        ).permute(0, 3, 1, 2).float()

        # flatten
        outputs = outputs.contiguous().view(outputs.shape[0], num_classes, -1)
        targets_onehot = targets_onehot.contiguous().view(
            targets_onehot.shape[0],
            num_classes,
            -1
        )

        intersection = (outputs * targets_onehot).sum(dim=2)

        dice = (2. * intersection + self.smooth) / (
            outputs.sum(dim=2) +
            targets_onehot.sum(dim=2) +
            self.smooth
        )

        dice = dice[:, 1:]
        dice_loss = 1 - dice.mean()

        return dice_loss


# FOCAL LOSS
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()

        self.alpha = alpha
        self.gamma = gamma

    def forward(self, outputs, targets):

        ce_loss = F.cross_entropy(
            outputs,
            targets,
            reduction='none',
            weight=self.alpha
        )

        pt = torch.exp(-ce_loss)

        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        return focal_loss.mean()


# DICE SCORE
def multiclass_dice_score(preds, targets, num_classes=5):

    preds = torch.argmax(preds, dim=1)
    dices = []

    for cls in range(1, num_classes):

        pred_cls = (preds == cls).float()
        target_cls = (targets == cls).float()

        intersection = (pred_cls * target_cls).sum()

        dice = (2 * intersection + 1e-5) / (pred_cls.sum() + target_cls.sum() + 1e-5)
        dices.append(dice.item())

    return sum(dices) / len(dices)

# IOU SCORE
def multiclass_iou_score(preds, targets, num_classes=5):

    preds = torch.argmax(preds, dim=1)

    ious = []

    for cls in range(1, num_classes):

        pred_cls = (preds == cls).float()
        target_cls = (targets == cls).float()

        intersection = (pred_cls * target_cls).sum()

        union = (
            pred_cls.sum() +
            target_cls.sum() -
            intersection
        )

        iou = (intersection + 1e-5) / (union + 1e-5)
        ious.append(iou.item())

    return sum(ious) / len(ious)


def train_model():
    print("Program started")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")

    dataset, train_loader, val_loader, test_loader = load_dataset()

    print(f"Total samples: {len(dataset)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # MODEL
    model = UNet().to(device)
    weights = torch.tensor([0.5, 1.0, 1.0, 2.0, 2.0]).to(device)

    focal_loss = FocalLoss(alpha=weights, gamma=2)

    dice_loss = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # TRAINING
    best_val_loss = float("inf")

    for epoch in range(35):
        print(f"--- Epoch {epoch + 1} started ---")

        model.train()
        train_loss = 0
        train_dice = 0
        train_iou = 0

        for batch_idx, (images, masks) in enumerate(train_loader):

            images = images.to(device)
            masks = masks.to(device)

            masks = masks.squeeze(1).long()

            if batch_idx == 0 and epoch == 0:
                print("Mask classes:", torch.unique(masks))

            outputs = model(images)

            loss_focal = focal_loss(outputs, masks)
            loss_dice = dice_loss(outputs, masks)

            loss = loss_focal + loss_dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            dice = multiclass_dice_score(outputs, masks)
            train_dice += dice

            iou = multiclass_iou_score(outputs, masks)
            train_iou += iou

            # простий лог батчів
            if batch_idx % 10 == 0:
                print(
                    f"Batch {batch_idx}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Dice: {dice:.4f} | "
                    f"IoU: {iou:.4f}"
                )

        train_loss /= len(train_loader)
        train_dice /= len(train_loader)
        train_iou /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        val_dice = 0
        val_iou = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                masks = masks.squeeze(1).long()

                outputs = model(images)

                loss_focal = focal_loss(outputs, masks)
                loss_dice = dice_loss(outputs, masks)

                loss = loss_focal + loss_dice

                val_loss += loss.item()

                dice = multiclass_dice_score(outputs, masks)
                val_dice += dice

                iou = multiclass_iou_score(outputs, masks)
                val_iou += iou

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        val_iou /= len(val_loader)

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Dice: {train_dice:.4f} | "
            f"Train IoU: {train_iou:.4f}"
        )

        print(
            f"Val Loss: {val_loss:.4f} | "
            f"Val Dice: {val_dice:.4f} | "
            f"Val IoU: {val_iou:.4f}"
        )

        # SAVE BEST MODEL
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(model.state_dict(), "best_unet_model_6.pth")

            print("Best model saved")

    print("Training finished")

    # TESTING
    print("\n=== TESTING ===")

    model.load_state_dict(torch.load("best_unet_model_6.pth"))
    model.eval()

    test_dice = 0
    test_iou = 0

    with torch.no_grad():

        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)

            masks = masks.squeeze(1).long()

            outputs = model(images)

            dice = multiclass_dice_score(outputs, masks)
            test_dice += dice

            iou = multiclass_iou_score(outputs, masks)
            test_iou += iou

    test_dice /= len(test_loader)
    test_iou /= len(test_loader)

    print(f"Final Test Dice Score: {test_dice:.4f}")
    print(f"Final Test IoU Score: {test_iou:.4f}")
