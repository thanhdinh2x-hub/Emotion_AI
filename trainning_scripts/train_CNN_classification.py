# đoạn code này để chạy bằng terminal
import os
import sys
from torch.utils.checkpoint import checkpoint

# thư mục của file hiện tại: ...\AI\Deeplearning\project3331\trainning_scripts
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# lên 3 cấp: trainning_scripts -> project3331 -> Deeplearning -> AI
AI_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))

if AI_DIR not in sys.path:
    sys.path.insert(0, AI_DIR)
# end chạy bằng terminal

import torch
from torch import nn
from Deeplearning.project3331.dataset.face_expression_dataset import  EmotionDataset
from Deeplearning.project3331.model.classification_model import SimpleCNN
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from torchvision import transforms
from argparse import ArgumentParser
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score  # thư viện tính các giá trị recall, accuracy,.. của confusion metrics
from torch.utils.tensorboard import SummaryWriter
import multiprocessing
import shutil
import matplotlib.pyplot as  plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np


def get_args():
    parser = ArgumentParser(description="CNN training script")
    parser.add_argument("--epochs", "-e", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch-size", "-b", type=int, default=8, help="number of batchs")
    parser.add_argument("--image-size", "-i", type=int, default=224, help="number of image-size")
    parser.add_argument("--root", "-r", type=str, default='../data/dataset', help="root of the dataset")
    parser.add_argument("--lst_path", "-lst", type=str, default='../data/emotion/label.lst', help="root of the labels of dataset")
    parser.add_argument("--logging", "-l", type=str, default="tensorboard")
    parser.add_argument("--trained-models", "-t", type=str, default="trained_models") # chứa các checkpoint
    parser.add_argument("--checkpoint", "-c", type=str, default=None) # chứa các checkpoint


    args = parser.parse_args()
    return args


def plot_confusion_matrix(writer, cm, class_names, epoch):
    """
    Phiên bản tối ưu cho Windows, dùng font mặc định.
    """
    # TỰ ĐỘNG PHÁT HIỆN HỆ ĐIỀU HÀNH
    import platform
    system = platform.system()

    # 1. THIẾT LẬP FONT THEO HỆ ĐIỀU HÀNH
    if system == 'Windows':
        # Font mặc định của Windows 10/11
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Segoe UI', 'Arial', 'Tahoma', 'Calibri']
    elif system == 'Darwin':  # macOS
        plt.rcParams['font.family'] = 'AppleGothic'
    else:  # Linux
        plt.rcParams['font.family'] = 'DejaVu Sans'

    plt.rcParams['axes.unicode_minus'] = False

    # 2. TẠO FIGURE
    n_classes = len(class_names)
    fig_size = max(10, n_classes * 0.7)  # Tự động điều chỉnh

    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.8))

    # 3. VẼ MATRIX
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Dùng matshow thay vì imshow để có nhiều tùy chọn hơn
    cax = ax.matshow(cm_normalized, cmap=plt.cm.Blues, alpha=0.8)

    # 4. THÊM COLORBAR
    plt.colorbar(cax, ax=ax)

    # 5. THÊM LABELS
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    # 6. XOAY VÀ CĂN CHỈNH NHÃN
    plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")

    # 7. THÊM GIÁ TRỊ
    for i in range(n_classes):
        for j in range(n_classes):
            value = f'{cm_normalized[i, j]:.2f}'
            ax.text(j, i, value,
                    ha='center', va='center',
                    color='white' if cm_normalized[i, j] > 0.5 else 'black',
                    fontsize=8)

    # 8. THÊM TITLE
    ax.set_title(f'Epoch {epoch} - Confusion Matrix', pad=20)

    # 9. TIGHT LAYOUT
    plt.tight_layout()

    # 10. ADD TO TENSORBOARD
    writer.add_figure('confusion_matrix', fig, epoch)

    plt.close(fig)
    # # TODO: thay thế plot_confusion_matrix(writer,confusion_matrix(all_labels,all_predictions),class_names=test_dataset.categories,epoch=epoch)
    # # emotion_map = {
    # #     0: "angry", 1: "disgust", 2: "fear",
    # #     3: "happy", 4: "sad", 5: "surprise", 6: "neutral"
    # # }
    # #  đổi thành tên thật, ví dụ:
    # class_names = ["angry","disgust","fear","happy","sad","surprise","neutral"]
    #
    # cm = confusion_matrix(all_labels, all_predictions)
    #
    # fig, ax = plt.subplots(figsize=(6, 6))
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    # disp.plot(ax=ax, cmap="Blues", colorbar=True)
    # plt.xticks(rotation=45)
    #
    # writer.add_figure("confusion_matrix", fig, global_step=epoch)
    # plt.close(fig)
    # # End TODO: thay thế plot_confusion_matrix(writer,confusion_matrix(all_labels,all_predictions),class_names=test_dataset.categories,epoch=epoch)

if __name__ == '__main__':
    # num_epochs = 1
    # batch_size = 8
    args = get_args()
    # check xem có GPU ngay từ đầu
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA available")
    else:
        device = torch.device("cpu")
    #end check xem có GPU ngay từ đầu

    transforms = Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])
    train_dataset = EmotionDataset(root_dir=args.root, mode='train', test_split=0.2, seed=42,
                                   transform=transforms)

    # Lấy số core dẩy hết vào dùng
    num_cores = multiprocessing.cpu_count()
    print(f"🎯 System has {num_cores} CPU cores")

    # chia data trong mỗi epochs
    training_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=num_cores-1,
        shuffle=True,  # trước khi generate ra ảnh thì tráo emotion lần đối với mẫu epochs
        drop_last=False,
    )
    test_dataset = EmotionDataset(root_dir=args.root, mode='test', test_split=0.2, seed=42,
                                   transform=transforms)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=num_cores-1,
        shuffle=True,
        drop_last=False,
    )# end chia data trong mỗi epochs

    if os.path.isdir(args.logging):
        shutil.rmtree(args.logging) # xoá hết  file
    if not os.path.isdir(args.trained_models): # check xem có trained_models ko để tạo
        os.mkdir(args.trained_models)

    writer = SummaryWriter(args.logging)

    # gọi model và các hàm để chuẩn bị tính toán
    model = SimpleCNN().to(device)


    criterion = nn.CrossEntropyLoss()  # gan ham tinh loss
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001,
                                momentum=0.9)  # model.parameters() là muốn update toàn bộ parmeter
    # end gọi model và các hàm để chuẩn bị tính toán

    if args.checkpoint:# nếu có checkpoint thì load hết cacs tham số cũ
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint['epoch']
        best_acc= checkpoint['best_acc']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        start_epoch = 0
        best_acc = 0

    num_interation = len(training_loader)  # so interation trong emotion epochs
    num_interation_test = len(test_loader)  # so interation trong emotion epochs

    # if torch.cuda.is_available():  # cach chuyen de chay tren GPU nhanh hon
    #     print("CUDA available")
    #     model.to(device)

    # train và evaluate với mỗi epochs
    for epoch in range(start_epoch,args.epochs):

        model.train()  # trước khi chạy phải chỉ ra chế dộ train- thực thi hành vi Dropout, BatchNorm để phù hợp
        progress_bar = tqdm(training_loader,colour="green") # thêm chạy  cho đẹp

        # train step
        for iter, (image, label) in enumerate(
                progress_bar):  # DataLoader ko có tự trả ra image,lable mà dataset.__getitem__ để lấy ra, còn dataloader chỉ là cái vỏ đọc đẻ chia batch thôi
            if torch.cuda.is_available():  # cach chuyen de chay tren GPU nhanh hon
                # print("CUDA cho image/label")
                image = image.to(device)
                label = label.to(device)
            # foward pass
            out_put = model(image)
            loss_value = criterion(out_put, label)
            if (iter + 1) % 10 == 0:
                progress_bar.set_description("Epochs: {}/{}. Iteration :{}/{}.Loss={:.3f}".format(epoch + 1, args.epochs, iter + 1,
                                                                       num_interation, loss_value)) # in theo progress_bar cho đẹp

            writer.add_scalar("Train/Loss", loss_value, epoch*num_interation+iter) # ghi lại loss trong bộ train vào tensorboard



            # backward
            optimizer.zero_grad()  # xóa hết bộ nhớ về gradient đi vì chưa cần lm vc với video
            loss_value.backward()  # dựa vào loss tính gradient
            optimizer.step()  # quay lại update lại pameters
        progress_bar.close()  # Đóng progress bar training
        # end train step

        # evaluation step
        progress_bar_test = tqdm(test_loader,colour="BLUE") # thêm chạy  cho đẹp

        model.eval()  # validate sau mỗi epochs- bỏ hết hành vi Dropout, BatchNorm để phù hợp “chế độ thi”.
        all_predictions = []
        all_labels = []
        for iter, (image, label) in enumerate(
                progress_bar_test):  # DataLoader ko có tự trả ra image,lable mà dataset.__getitem__ để lấy ra, còn dataloader chỉ là cái vỏ đọc đẻ chia batch thôi
            all_labels.extend(label)
            if torch.cuda.is_available():  # cach chuyen de chay tren GPU nhanh hon
                # print("CUDA cho image/label ở validation")
                image = image.to(device)     # là tensor 4 chiều image.shape == [bathch_size, 3, 224, 224]
                label = label.to(device)
            # not backward
            with torch.no_grad():  # tất cả câu lệnh trọng câu lệnh này thì sẽ khôgn tính gradient để update model
                predictions = model(
                    image)  # kết quả sẽ là vetor 7 ptu, prediction shape [batch_size x 7], và kết quả là emotion tensor
                # print(predictions)
                values, indices = torch.max(predictions.to(device),
                                            dim=1)  # chỉ ra sô lớn nhất và index của số đó trong từng bức ảnh (64 bức ảnh theo batch_size và mỗi size chứa 10 ptu kết quả)
                all_predictions.extend(
                    indices)  # kết quả sẽ là mảng tensor vì input(image) ngay từ đầu đã l emotion tensor r
                loss_value_test = criterion(predictions, label)
                progress_bar_test.set_description("Epochs_Evaluate: {}/{}. Iteration :{}/{}.Loss_test={:.3f}".format(epoch + 1, args.epochs, iter + 1,
                                                                       num_interation_test, loss_value_test)) # in theo progress_bar cho đẹp
        print("------------------------------------------------------------------------------------------")

        # print(all_labels)
        print("------------------------------------------------------------------------------------------")
        # print(all_predictions)
        all_predictions = [prediction.item() for prediction in
                           all_predictions]  # do các phần từ trong metrics đang ở dạng tensor hết nên muốn lấy ra thì .item()
        all_labels = [label.item() for label in
                      all_labels]  # do các phần từ trong metrics đang ở dạng tensor hết nên muốn lấy ra thì .item()
        # print(all_labels)
        # TODO: dùng hàm tự chế để thêm confusion matrix vào tensorboard
        plot_confusion_matrix(writer,confusion_matrix(all_labels,all_predictions),class_names=test_dataset.categories,epoch=epoch)
        # End  TODO: dùng hàm tu chế để thêm confusion matrix vào tensorboard

        # print("------------------------------------------------------------------------------------------")
        # print(all_predictions)
        accuracy=accuracy_score(all_labels, all_predictions)
        print("Epoch :{} .Accuracy:{}.".format(epoch + 1,accuracy))

        writer.add_scalar("Val/Accuracy ", accuracy,epoch)
        # torch.save(model.state_dict(), "{}/last_cnn.pt".format(args.trained_models))

        checkpoint = {  # cái này để train epoch tiếp tục khi ngày hôm qua dừng
            "epoch": epoch + 1,  # tại vì nay train xang đến epoch 50 r thì mai phải train từ 51
            "best_acc": best_acc, # ví dụ chạy hết epoch 2 và đang chạy epoch 3 mà thoaát ra thì lúc chạy lại thì nó lưu best_acc của epoch trước
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(checkpoint, "{}/last_cnn.pt".format(args.trained_models))

        if accuracy > best_acc:
            best_acc = accuracy
            checkpoint = {
                "epoch": epoch + 1,  # tại vì nay train xang đến epoch 50 r thì mai phải train từ 51
                "best_acc": best_acc,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(checkpoint, "{}/best_cnn.pt".format(args.trained_models))




        # print(classification_report( all_labels,all_predictions))
