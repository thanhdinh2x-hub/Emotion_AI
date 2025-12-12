from argparse import ArgumentParser
import numpy as np

import torch
import torch.nn as nn
import cv2

from Deeplearning.project3331.model.classification_model import SimpleCNN


def get_args():
    parser = ArgumentParser(description="CNN inference")
    parser.add_argument("--image-path", "-p", type=str, default=None)
    parser.add_argument("--image-size", "-i", type=int, default=224, help="number of image-size")

    parser.add_argument("--checkpoint", "-c", type=str, default=" trained_models/best_cnn.pt") # chứa các checkpoint

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    categories = ['Ahegao', 'Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

    args = get_args()
    # check xem có GPU ngay từ đầu
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA available")
    else:
        device = torch.device("cpu")
    # end check xem có GPU ngay từ đầu
    model=SimpleCNN(num_class=6).to(device)
    if args.checkpoint:
        checkpoint=torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model'])
    else:
        print("No checkpoint provided")
        exit(0)

    model.eval()

    ori_image = cv2.imread(args.image_path)
    image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (args.image_size, args.image_size))
    image = np.transpose(image, (2, 0, 1)) / 255.0
    image = image[None, :, :, :]  # 1 x 3 x 224 x 224
    image = torch.from_numpy(image).to(device).float()
    softmax = nn.Softmax()
    with torch.no_grad():
        output = model(image)
        probs = softmax(output)

    max_idx = torch.argmax(probs)
    predicted_class = categories[max_idx]
    print("The test image is about {} with confident score of {}".format(predicted_class, probs[0, max_idx]))
    cv2.imshow("{}:{:.2f}%".format(predicted_class, probs[0, max_idx] * 100), ori_image)
    cv2.waitKey(0)