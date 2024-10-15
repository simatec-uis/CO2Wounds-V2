import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import albumentations as albu
import torch
import numpy as np
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils.train import ValidEpoch
import segmentation_models_pytorch.utils
from torch.optim.lr_scheduler import StepLR

# helper function for data visualization
def visualize(fig_name=None, result=False, **images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        if result and i>0:
            plt.imshow(image, cmap='gray')
        else:
            plt.imshow(image)
    # plt.show()
    if fig_name is None:
        fig_name = "tmp.png"
    plt.savefig(fig_name)

class Dataset(BaseDataset):
    """CO2wounds Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    CLASSES = ['background', 'wound']
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks = [os.path.join(masks_dir, image_id.split(".")[0]+".png") for image_id in self.ids]
        
        # convert str names to class values on masks 
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks[i], 0)
        
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.ids)
    

def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        albu.RandomCrop(height=320, width=320, always_apply=True),

        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)


# def get_validation_augmentation():
#     """Add paddings to make image shape divisible by 32"""
#     test_transform = [
#         albu.PadIfNeeded(384, 480, )
#     ]
#     return albu.Compose(test_transform)
def get_validation_augmentation():
    """Adaptively add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.LongestMaxSize(max_size=384, always_apply=True),
        albu.PadIfNeeded(min_height=384, min_width=480, always_apply=True, border_mode=0, value=0),
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)



def main():
    ########### Load data and visualize ##################
    DATA_DIR = './data/CO2wounds/'

    # load repo with data if it is not exists
    if not os.path.exists(DATA_DIR):
        print('Please download our dataset')


    x_train_dir = os.path.join(DATA_DIR, 'train')
    y_train_dir = os.path.join(DATA_DIR, 'train_anns')

    x_valid_dir = os.path.join(DATA_DIR, 'val')
    y_valid_dir = os.path.join(DATA_DIR, 'val_anns')

    # x_test_dir = os.path.join(DATA_DIR, 'test')
    # y_test_dir = os.path.join(DATA_DIR, 'test_anns')

    # Lets look at data we have

    dataset = Dataset(x_train_dir, y_train_dir, classes=['wound'])

    image, mask = dataset[4] # get some sample 
    visualize(
        "initial_vis.png",
        result=False,
        image=image, 
        mask=mask.squeeze(),
    )

    ###########  Visualize resulted augmented images and masks ########### 

    augmented_dataset = Dataset(
        x_train_dir, 
        y_train_dir, 
        augmentation=get_training_augmentation(), 
        classes=['wound'],
    )

    # same image with different random transforms
    for i in range(3):
        image, mask = augmented_dataset[1]
        visualize("augm_"+str(i)+".png", result=False, image=image, mask=mask.squeeze(-1))


    ########### Create model and train ###########
    ENCODER = 'mit_b5' #'se_resnext50_32x4d'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = ['wound']
    ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation
    DEVICE = 'cuda'

    # create segmentation model with pretrained encoder. Replace the model here with the architecture of your choice.
    model = smp.FPN(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        classes=len(CLASSES), 
        activation=ACTIVATION,
    )
    # model = smp.DeepLabV3(
    #     encoder_name=ENCODER, 
    #     encoder_weights=ENCODER_WEIGHTS, 
    #     classes=len(CLASSES), 
    #     activation=ACTIVATION,
    # )

    # model = smp.DeepLabV3Plus(
    #     encoder_name=ENCODER, 
    #     encoder_weights=ENCODER_WEIGHTS, 
    #     classes=len(CLASSES), 
    #     activation=ACTIVATION,
    # )

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    train_dataset = Dataset(
    x_train_dir, 
    y_train_dir, 
    augmentation=get_training_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
    )

    valid_dataset = Dataset(
        x_valid_dir, 
        y_valid_dir, 
        augmentation=get_validation_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=12)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

    # Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    # IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index

    loss = smp.utils.losses.DiceLoss()
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
        smp.utils.metrics.Fscore()
    ]

    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=0.0001),
    ])

    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    starting_epoch_scheduler=40 # TODO: tune this hyperparameter

    # define number of epochs
    max_score = 0
    num_epochs = 100 # TODO: tune this hyperparameter

    # create epoch runners 
    # it is a simple loop of iterating over dataloader`s samples
    train_epoch = smp.utils.train.TrainEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        device=DEVICE,
        verbose=True,
    )

   

    for i in range(0, num_epochs): # num_epochs
        
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        if i >= starting_epoch_scheduler:
            # Update the learning rate
            scheduler.step()
        
        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, './best_checkpoint.pth')
            print('Best model checkpoint saved!')
            
        if i == 60: # TODO: tune this hyperparameter  (before 25)
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')

        if i == num_epochs - 1:
            torch.save(model, './last_checkpoint.pth')
            print('Last checkpoint saved!')

    ########### Test best saved model ###########
    # load best saved checkpoint
    best_model = torch.load('./best_checkpoint.pth')
    # # create test dataset
    # test_dataset = Dataset(
    #     x_test_dir, 
    #     y_test_dir, 
    #     augmentation=get_validation_augmentation(), 
    #     preprocessing=get_preprocessing(preprocessing_fn),
    #     classes=CLASSES,
    # )


    test_dataloader = DataLoader(valid_dataset)
    # evaluate model on test set
    metrics_test= metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
        smp.utils.metrics.Fscore(),
        smp.utils.metrics.Accuracy(),
        smp.utils.metrics.Precision(),
        smp.utils.metrics.Recall()
    ]
    test_epoch = ValidEpoch(
        model=best_model,
        loss=loss,
        metrics=metrics_test,
        device=DEVICE,
        verbose=True,
    )

    logs = test_epoch.run(test_dataloader)

    ########### Visualize predictions ###########
    # test dataset without transformations for image visualization
    test_dataset_vis = Dataset(
        x_valid_dir, y_valid_dir, 
        classes=CLASSES,
    )

    for i in range(5):
        n = np.random.choice(len(valid_dataset))
        
        image_vis = test_dataset_vis[n][0].astype('uint8')
        image, gt_mask = valid_dataset[n]
        
        gt_mask = gt_mask.squeeze()
        
        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pr_mask = best_model.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())
            
        visualize(
            "result_"+str(i)+".png",
            result=True,
            image=image_vis, 
            ground_truth_mask=gt_mask, 
            predicted_mask=pr_mask
        )
    

if __name__ == "__main__":
    main()
