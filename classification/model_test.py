import torch
import torch.utils.data as Data
from torchvision import transforms
from ViT_model import CustomViT
from torchvision.datasets import ImageFolder
from PIL import Image
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm


def test_data_process():
    # 定义数据集的路径
    ROOT_TRAIN = 'classify_data/test'

    normalize = transforms.Normalize([0.0717982, 0.00635558, 0.01404482], [0.02460866, 0.00014787, 0.00068004])
    # 定义数据预处理操作
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    # 加载数据集
    test_data = ImageFolder(ROOT_TRAIN, transform=test_transform)

    test_dataloader = Data.DataLoader(dataset=test_data,
                                      batch_size=12,
                                      shuffle=True,
                                      num_workers=10)
    return test_dataloader


def test_model_process(model, test_dataloader):
    device = "cuda" if torch.cuda.is_available() else 'cpu'

    # 讲模型放入到训练设备中
    model = model.to(device)

    # 初始化参数
    test_corrects = 0.0
    test_num = 0

    # Prepare to track the predictions
    all_preds = []
    all_targets = []

    # Perform forward propagation only, without calculating gradients to save memory and increase speed
    with torch.no_grad():
        for test_data_x, test_data_y in tqdm(test_dataloader):
            # Move features to the testing device
            test_data_x = test_data_x.to(device)
            # Move labels to the testing device
            test_data_y = test_data_y.to(device)
            # Set model to evaluation mode
            model.eval()
            # Forward propagation: input the test dataset, output predictions for each sample
            output = model(test_data_x)
            # Find the index of the maximum value in each row
            pre_lab = torch.argmax(output, dim=1)
            # If the prediction is correct, increment test_corrects by 1
            test_corrects += torch.sum(pre_lab == test_data_y.data)
            # Accumulate the count of all test samples
            test_num += test_data_x.size(0)

            all_preds.extend(pre_lab.view(-1).cpu().numpy())
            all_targets.extend(test_data_y.cpu().numpy())
    # Calculate test accuracy
    test_acc = test_corrects.double().item() / test_num
    print("Test accuracy:", test_acc)

    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(1, 9), yticklabels=range(1, 9))
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()


if __name__ == "__main__":
    # Load model
    model = CustomViT(8)
    model.load_state_dict(torch.load('best_model_ViT_8.pth'))
    # Test the model using the existing model setup
    test_dataloader = test_data_process()
    test_model_process(model, test_dataloader)


   
