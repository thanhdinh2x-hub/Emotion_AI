import torch
from torch import nn
from Deeplearning.project3331.dataset.dataset import FaceClassificationDataset
from Deeplearning.project3331.model.classification_model import SimpleCNN
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from torchvision import transforms

if __name__ == '__main__':
    num_epochs = 1
    bathch_size = 8
    transforms = Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    train_dataset = FaceClassificationDataset(root='../data/emotion', lst_path='../data/emotion/label.lst', mode='train',
                                              test_size=0.2, random_state=42, transform=transforms)
    training_loader = DataLoader(
        dataset=train_dataset,
        batch_size=bathch_size,
        num_workers=4,
        shuffle=True,  # trước khi generate ra ảnh thì tráo emotion lần đối với mẫu epochs
        drop_last=False,
    )
    test_dataset = FaceClassificationDataset(root='../data/emotion', lst_path='../data/emotion/label.lst', mode='test', test_size=0.2,
                                             random_state=42, transform=transforms)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=bathch_size,
        num_workers=4,
        shuffle=True,
        drop_last=False,
    )

    model = SimpleCNN(num_class=7)
    criterion = nn.CrossEntropyLoss()  # gan ham tinh loss
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001,
                                momentum=0.9)  # model.parameters() là muốn update toàn bộ parmeter
    num_interation = len(training_loader)  # so interation trong emotion epochs
    if torch.cuda.is_available():  # cach chuyen de chay tren GPU nhanh hon
        model.cuda()
    for epoch in range(num_epochs):
        model.train()  # trước khi chạy phải chỉ ra tôi dùng model này để train
        # train step
        for iter, (image, label) in enumerate(
                training_loader):  # DataLoader ko có tự trả ra image,lable mà dataset.__getitem__ để lấy ra, còn dataloader chỉ là cái vỏ đọc đẻ chia batch thôi
            if torch.cuda.is_available():  # cach chuyen de chay tren GPU nhanh hon
                image = image.cuda()
                label = label.cuda()
            # foward pass
            out_put = model(image)
            loss_value = criterion(out_put, label)
            if (iter + 1) % 10 == 0:
                print("Epochs: {}/{}.\n Iteration :{}/{}.\nLoss={}".format(epoch + 1, num_epochs, iter + 1,
                                                                           num_interation, loss_value))
            # backward
            optimizer.zero_grad()  # xóa hết bộ nhớ về gradient đi vì chưa cần lm vc với video
            loss_value.backward()  # dựa vào loss tính gradient
            optimizer.step()  # quay lại update lại pameters

        model.eval()  # validate sau mỗi epochs
        all_predictions = []
        all_labels = []
        for iter, (image, label) in enumerate(
                training_loader):  # DataLoader ko có tự trả ra image,lable mà dataset.__getitem__ để lấy ra, còn dataloader chỉ là cái vỏ đọc đẻ chia batch thôi
            all_labels.extend(label)
            if torch.cuda.is_available():  # cach chuyen de chay tren GPU nhanh hon
                image = image.cuda()
                label = label.cuda()
            with torch.no_grad():  # tất cả câu lệnh trọng câu lệnh này thì sẽ khôgn tính gradient để update model
                predictions = model(
                    image)  # kết quả sẽ là vetor 10 ptu, prediction shape [batch_size x 10], và kết quả là emotion tensor
                print(predictions)
                values, indices = torch.max(predictions.cpu(),
                                            dim=1)  # chỉ ra sô lớn nhất và index của số đó trong từng bức ảnh (64 bức ảnh theo batch_size và mỗi size chứa 10 ptu kết quả)
                all_predictions.extend(indices)  # kết quả sẽ là mảng tensor vì input(image) ngay từ đầu đã l emotion tensor r
                loss_value = criterion(predictions, label)
        print("------------------------------------------------------------------------------------------")
        print(all_labels)
        print("------------------------------------------------------------------------------------------")
        print(all_predictions)
        all_predictions = [prediction.item() for prediction in
                           all_predictions]  # do các phần từ trong metrics đang ở dạng tensor hết nên muốn lấy ra thì .item()
        all_labels = [label.item() for label in
                      all_labels]  # do các phần từ trong metrics đang ở dạng tensor hết nên muốn lấy ra thì .item()
        print(all_labels)
        print("------------------------------------------------------------------------------------------")
        print(all_predictions)
