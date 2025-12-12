import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, Dict, List
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.model_selection import train_test_split
from PIL import Image
from torchvision import transforms
from torchvision.transforms import ToTensor , Resize , Compose,ToPILImage


import numpy as np

class EmotionDataset(Dataset):
    """
    Dataset cho bài toán Facial Expression Recognition (FER).
    Dataset được tổ chức theo cấu trúc folder.
    Chỉ có 2 mode: 'train' và 'test'

    Attributes:
        root_dir (str): Đường dẫn đến thư mục gốc chứa dataset
        mode (str): Chế độ của dataset ('train', 'test')
        transform (Callable): Các phép biến đổi áp dụng lên ảnh
        classes (List[str]): Danh sách tên các lớp (biểu cảm)
        class_to_idx (Dict[str, int]): Ánh xạ từ tên lớp sang chỉ số
        image_paths (List[str]): Danh sách đường dẫn đến tất cả ảnh
        labels (List[int]): Danh sách nhãn tương ứng với tất cả ảnh
        selected_paths (List[str]): Đường dẫn ảnh được chọn theo mode
        selected_labels (List[int]): Nhãn tương ứng được chọn theo mode
    """
    def load_all_images(self):
        """
        Tải tất cả đường dẫn ảnh và nhãn từ cấu trúc thư mục.
        """
        # Duyệt qua từng lớp (biểu cảm)
        for emotion in self.categories:
            emotion_dir = os.path.join(self.root_dir, emotion)

            # Kiểm tra nếu thư mục tồn tại
            if not os.path.isdir(emotion_dir):
                continue

            # Lấy tất cả file ảnh trong thư mục lớp
            for filename in os.listdir(emotion_dir):
                # Kiểm tra đuôi file ảnh
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    img_path = os.path.join(emotion_dir, filename)
                    self.all_image_paths.append(img_path)
                    self.all_labels.append(self.class_to_idx[emotion])

    def __init__(self,
                 root_dir: str,
                 mode: str = 'train',
                 transform= None,
                 test_split: float = 0.2,
                 seed: int = 42,
                 stratify: bool = True):
        """
        Khởi tạo dataset và tự động chia thành train/test.

        Args:
            root_dir: Đường dẫn đến thư mục gốc chứa dataset
            mode: Chế độ của dataset ('train', 'test')
            transform: Các phép biến đổi áp dụng lên ảnh
            test_split: Tỷ lệ dữ liệu test (mặc định 20%)
            seed: Random seed để đảm bảo reproducibility
            stratify: Nếu True, chia dữ liệu giữ nguyên phân bố các lớp
        """

        # --- BƯỚC 1: KHỞI TẠO CÁC THUỘC TÍNH CƠ BẢN ---
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.seed = seed
        self.test_split = test_split

        # Kiểm tra mode có hợp lệ không
        if mode not in ['train', 'test']:
            raise ValueError(f"Mode '{mode}' không hợp lệ. Chỉ chọn 'train' hoặc 'test'")

        # Kiểm tra tỷ lệ test
        if test_split <= 0 or test_split >= 1:
            raise ValueError("test_split phải nằm trong khoảng (0, 1)")

        # --- BƯỚC 2: TẢI THÔNG TIN LỚP TỪ CẤU TRÚC THƯ MỤC ---
        # Lấy tất cả item trong thư mục gốc
        all_items = os.listdir(root_dir)

        # Khởi tạo list rỗng để chứa tên các folder (lớp)
        class_folders = []

        # Duyệt qua từng item
        for item in all_items:
            # Tạo đường dẫn đầy đủ
            item_path = os.path.join(root_dir, item)

            # Kiểm tra nếu là thư mục thì thêm vào list
            if os.path.isdir(item_path):
                class_folders.append(item)


        # Lưu vào biến instance
        self.categories = class_folders # TODO:['Ahegao', 'Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

        # Tạo mapping từ tên lớp -> số (ví dụ: 'angry' -> 0)
        self.class_to_idx = {}
        # Tạo mapping từ số -> tên lớp (ví dụ: 0 -> 'angry')
        self.idx_to_class = {}

        # Duyệt qua từng lớp, gán chỉ số tăng dần từ 0
        for index, class_name in enumerate(self.categories):
            self.class_to_idx[class_name] = index  # 'angry' -> 0
            self.idx_to_class[index] = class_name  # 0 -> 'angry'

        print(f"Classes found: {self.categories}")
        print(f"Class mapping: {self.class_to_idx}")
        print(f"Index mapping: {self.idx_to_class}")

        # --- BƯỚC 3: TẢI TẤT CẢ ĐƯỜNG DẪN ẢNH VÀ NHÃN ---
        self.all_image_paths = []  # Lưu đường dẫn đầy đủ đến ảnh
        self.all_labels = []  # Lưu nhãn tương ứng (dạng số)

        self.load_all_images() # load hết toàn bộ ảnh và label thành 2 mảng

        # print(f"All images loaded: {self.all_image_paths[0]}")
        # Kiểm tra xem có ảnh nào không
        if len(self.all_image_paths) == 0:
            raise RuntimeError(f"Không tìm thấy ảnh nào trong {root_dir}")

        # --- BƯỚC 4: CHIA DATASET THÀNH TRAIN/TEST ---
        # Chia dữ liệu thành train và test
        if  mode in ['train', 'test']:
            x_train, x_test, y_train, y_test = train_test_split(
                self.all_image_paths, self.all_labels,
                test_size=test_split,
                random_state=self.seed,
                stratify=self.all_labels  # để dữ liệu cân bằng giữa train và test
            )
            self.images_path = []
            self.labels = []
            if mode == 'train':  # nếu mode là train thì cho xem bộ train
                self.images_path=x_train
                self.labels = y_train
            else:  # mode == 'test' # nếu mode là test thì cho xem bộ test
                self.images_path = x_test
                self.labels = y_test
        else:  # mode == 'all' hoặc test_size=0
            # Cần xử lý cho mode 'all'
            self.images_path =self.all_image_paths
            self.labels = self.all_labels



    def __len__(self):
        """
        Trả về số lượng mẫu trong dataset.
        """
        return len(self.labels)

    def __getitem__(self, index):
        """
        Lấy một mẫu (ảnh, nhãn) từ dataset.

        Args:
            idx: Chỉ số của mẫu cần lấy

        Returns:
            Tuple[torch.Tensor, int]: (ảnh đã transform, nhãn)
        """
        image_path = self.images_path[index]
        label = self.labels[index]
        image = image_path
        if self.transform:
            image = Image.open(image).convert(
                "RGB")  # tạo emotion ảnh PIL và ép ảnh thành ảnh RGB có 3 chiều (bất kể là ảnh gì thì cũng bị ép)
            image = self.transform(image)  # trả về ảnh dướng dạng tensor
        # # đọc ảnh bằng PIL
        #
        # image_show= Image.open(image_path) # ko convert sang pixel đc
        # image_show.show()
        return image, label

    @staticmethod
    def get_emotion_name(label):
        emotion_map = {
            0: 'Ahegao', 1: 'Angry', 2: 'Happy', 3: 'Neutral', 4: 'Sad', 5: 'Surprise'
        }
        return emotion_map.get(label, f"unknown_{label}")
        # def get_emotion_name(self, label):
        #     """Khi cần show tên biểu cảm thì gọi hàm này"""
        #     return self.EMOTION_NAMES.get(label, f"unknown_{label}")


if __name__ == '__main__':
    transforms = Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    train_dataset = EmotionDataset(root_dir='../data/dataset', mode='train', test_split=0.2, seed=42, transform=transforms)
    # image= dataset[0]
    # print(image)
    test_dataset = EmotionDataset(root_dir='../data/dataset', mode='test', test_split=0.2, seed=42, transform=transforms)

    import numpy as np

    print(np.unique(train_dataset.labels))
    print("Số class:", len(np.unique(train_dataset.labels)))
    image,label=train_dataset.__getitem__(100)
    emotion= train_dataset.get_emotion_name(label)
    print(image.shape)
    print(emotion)
    length_train= train_dataset.__len__()
    print("length_train:", length_train)
    length_test= test_dataset.__len__()
    print("length_test:", length_test)



    # if os.path.exists(path):
    #     print(f"✅ Tồn tại: {path}")
    # else:
    #     print(f"❌ Không tồn tại: {path}")
    #     print(f"   Thư mục hiện tại: {os.getcwd()}")
    #     print(f"   Đường dẫn tuyệt đối: {os.path.abspath(path)}")
    epochs = 10

    training_loader = DataLoader(
        dataset=train_dataset,
        batch_size=2,
        num_workers=4,
        shuffle=True,  # trước khi generate ra ảnh thì tráo emotion lần đối với mẫu epochs
        drop_last=False,
    )
    print(
        training_loader)  # chả ra cái gì ấy chả liên quan trả mỗi <torch.utils.data.dataloader.DataLoader object at 0x000002284D014AD0>
    for epoch in range(epochs):
        for images, labels in training_loader:
            print(images, labels)
            break
        print('epochs thứ {}:'.format(epoch))

