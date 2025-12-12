from torch.utils.data import Dataset, DataLoader
import os
from sklearn.model_selection import train_test_split
from PIL import Image
from torchvision import transforms
from torchvision.transforms import ToTensor , Resize , Compose,ToPILImage


import numpy as np


def parse_lst_to_file(lst_path): # hàm này lấy ra mảng tên ảnh và labels

    image_names = []
    labels = []

    with open(lst_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8:
                image_name = parts[0]
                label = int(parts[7])
                image_names.append(image_name)
                labels.append(label)
        # print(image_names)
        # print(labels)
    return image_names, labels


class FaceClassificationDataset(Dataset):
    def __init__(self, root, lst_path, mode='train', test_size=0.2,
                 random_state=42, transform=None):

        """
        Args:
            root_dir: Thư mục chứa ảnh
            lst_path: Đường dẫn file label.lst
            mode: 'train', 'test', hoặc 'all'
            test_size: Tỷ lệ test
            random_state: giữ cho bộ train v test không đổi
            transform: Transform áp dụng
            image_dir: đường dẫn file chứa tất cả ảnh
        """
        self.root = root
        self.mode = mode
        self.categories=[["angry","disgust","fear","happy","sad","surprise","neutral"]] #TODO : nhớ check kỹ label xem đúng thứ tự chưa
        self.transform = transform
        self.image_dir = os.path.join(root, 'origin')

        # Đọc toàn bộ dữ liệu
        all_image_names, all_labels = parse_lst_to_file(lst_path)

        # Chia train/test
        if test_size > 0 and mode in ['train', 'test']:
            x_train, x_test, y_train, y_test = train_test_split(
                all_image_names, all_labels,
                test_size=test_size,
                random_state=random_state,
                stratify=all_labels # để dữ liệu cân bằng giữa train và test
            )
            self.images_path = []
            self.labels = []
            if mode == 'train': # nếu mode là train thì cho xem bộ train
                image_names_array_train = x_train
                for image_name in image_names_array_train:
                    file_path = os.path.join(self.image_dir, image_name)
                    self.images_path.append(file_path) # trả về emotion mảng chứa các đường dẫn

                self.labels = y_train
            else:  # mode == 'test' # nếu mode là test thì cho xem bộ test
                image_names_array_test = x_test
                for image_name in image_names_array_test:
                    file_path = os.path.join(self.image_dir, image_name)
                    self.images_path.append(file_path) # trả về emotion mảng chứa các đường dẫn

                self.labels = y_test
        else:  # mode == 'all' hoặc test_size=0
            # Cần xử lý cho mode 'all'
            self.images_path = []
            self.labels = all_labels
            for image_name in all_image_names:
                file_path = os.path.join(self.image_dir, image_name)
                self.images_path.append(file_path)

        # print(f"Dataset '{mode}': {len(image_names)} samples")
        # print(self.images_path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image_path = self.images_path[index]
        label = self.labels[index]
        image= image_path
        if self.transform:
            image = Image.open(image).convert("RGB")  # tạo emotion ảnh PIL và ép ảnh thành ảnh RGB có 3 chiều (bất kể là ảnh gì thì cũng bị ép)
            image = self.transform(image) # trả về ảnh dướng dạng tensor
        # đọc ảnh bằng PIL
        #
        # image_show= Image.open(image_path) # ko convert sang pixel đc
        # image_show.show()
        return image, label

    @ staticmethod
    def get_emotion_name(label):
        emotion_map = {
            0: "angry", 1: "disgust", 2: "fear",
            3: "happy", 4: "sad", 5: "surprise", 6: "neutral"
        }
        return emotion_map.get(label, f"unknown_{label}")
    # def get_emotion_name(self, label):
    #     """Khi cần show tên biểu cảm thì gọi hàm này"""
    #     return self.EMOTION_NAMES.get(label, f"unknown_{label}")

if __name__ == '__main__':
    # dataset = FaceClassificationDataset(root='../data/emotion', lst_path='../data/emotion/label.lst', mode='train', test_size=0.2, random_state=42, transform=None)
    #
    # image = dataset[0]
    # print(image)
    transforms = Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    dataset = FaceClassificationDataset(root='../data/emotion', lst_path='../data/emotion/label.lst', mode='train', test_size=0.2, random_state=42, transform=transforms)
    import numpy as np

    print(np.unique(dataset.labels))
    print("Số class:", len(np.unique(dataset.labels)))
    image,label=dataset.__getitem__(100)
    emotion= dataset.get_emotion_name(label)
    print(image.shape)
    print(emotion)
    dataset.__len__()



    # path = "../data/emotion"
    #
    # if os.path.exists(path):
    #     print(f"✅ Tồn tại: {path}")
    # else:
    #     print(f"❌ Không tồn tại: {path}")
    #     print(f"   Thư mục hiện tại: {os.getcwd()}")
    #     print(f"   Đường dẫn tuyệt đối: {os.path.abspath(path)}")
    # epochs = emotion
    #
    # training_loader = DataLoader(
    #     dataset=dataset,
    #     batch_size=2,
    #     num_workers=4,
    #     shuffle=True,  # trước khi generate ra ảnh thì tráo emotion lần đối với mẫu epochs
    #     drop_last=False,
    # )
    # print(
    #     training_loader)  # chả ra cái gì ấy chả liên quan trả mỗi <torch.utils.data.dataloader.DataLoader object at 0x000002284D014AD0>
    # for epoch in range(epochs):
    #     for images, labels in training_loader:
    #         print(images, labels)
    #         break
    #     print('epochs thứ {}:'.format(epoch))

