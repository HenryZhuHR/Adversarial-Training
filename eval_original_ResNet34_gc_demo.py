import torch
from torchvision import transforms, datasets
from unfolded_resnet34 import ResNet34


def main():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # test_dataset = datasets.ImageFolder(root="/workspace/yjt/gc10_dsets/test224",
    #                                         transform=transforms.ToTensor())
    # test_dataset = datasets.ImageFolder(root="/workspace/yjt/gc10_dsets/pgd_8_255_demo",
    #                                         transform=transforms.ToTensor())
    test_dataset = datasets.ImageFolder(root="E:/datasets/gc10_dsets/recon_pgd_8_255_demo",
                                            transform=transforms.ToTensor())

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=1, shuffle=False,
                                                  num_workers=0)
    model = ResNet34()
    model.to(device)
    test_num = 2000
    model_weight_path = "./9_100epoch_t1_resNet34_32_3e-4.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=torch.device('cpu')))
    model.eval()
    acc = 0.00
    print("\n\n\n")
    with torch.no_grad():
        for test_images, test_labels in test_loader:
            outputs = model(test_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += (predict_y == test_labels.to(device)).sum().item()
            if predict_y == test_labels.to(device):
                print("1")
            else:
                print("0")
        test_accurate = 100.0 * acc / test_num
    print("\n\n\nacc_num:",acc)
    print("test_num",test_num)
    print("test_accurate：",test_accurate,"%")


if __name__ == '__main__':
    main()
