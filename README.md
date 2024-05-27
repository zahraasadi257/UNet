
# UNet for Image Segmentation

Welcome to the repository for our UNet-based image segmentation model, implemented in PyTorch. This model is designed to perform semantic segmentation on the Carvana Image Masking Challenge dataset, which can be found on Kaggle. The UNet architecture is a powerful convolutional network for biomedial image segmentation, and we've adapted it for car image segmentation.

،This implementation has been inspired by [this video](https://www.youtube.com/watch?v=HS3Q_90hnDg&t=10s)

## Model Architecture

The `UNet` class defines the architecture of our model. It consists of four downsampling blocks (`DownSample`), a bottleneck, and four upsampling blocks (`UpSample`). Each block is composed of two convolutional layers followed by a ReLU activation function. The downsampling blocks also include max pooling, while the upsampling blocks use transposed convolutions for upsampling.







![UNet](https://github.com/zahraasadi257/UNet/assets/57061013/c1a63dca-80ac-48c5-99ee-b403a5da79f5)








## Dataset

The `CarvanaDataset` class handles the loading and preprocessing of the dataset. It downloads the data from the provided URL, resizes the images to 512x512 pixels, and converts them to PyTorch tensors.

## Training

The training loop runs for a specified number of epochs, using the Adam optimizer and Binary Cross-Entropy with Logits as the loss function. The model's performance is evaluated on both the training and validation datasets, with the losses printed at the end of each epoch.

## Usage

To train the model, simply run the provided code. Make sure to replace `DATA_URL` with the actual URL of the Carvana dataset and `MODEL_SAVE_PATH` with the desired path to save the trained model.

## Requirements

•  PyTorch

•  torchvision

•  tqdm

•  PIL

•  numpy

