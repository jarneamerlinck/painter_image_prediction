# Conv bases
## Summary

| Model        | Train_acc | Val_acc | Test_acc | Training Time (in minutes) |
| :----------- | :-------- | :------ | :------- | :------------------------- |
| Baseline     | 1         | 0.737   | 0.78     | 1.7                        |
| vgg16        | 0.999     | 0.922   | 0.969    | 3.2                        |
| vgg19        | 1         | 0.957   | 0.988    | 4.3                        |
| ResNet152    | 0.998     | 0.933   | 0.973    | 5.3                        |
| ResNet152_V2 | 0.998     | 0.937   | 0.969    | 5.1                        |
| Xception     | 0.999     | 0.894   | 0.949    | 2.4                        |



## vgg16
### Summary

| Model                             | Train_acc | Val_acc | Test_acc | Training Time (in minutes) |
| :-------------------------------- | :-------- | :------ | :------- | :------------------------- |
| conv base fully trainable         | 0.379     | 0.392   | 0.392    | 9.2                        |
| conv base not trainable           | 0.999     | 0.925   | 0.965    | 3.1                        |
| conv base last 4 layers trainable | 0.387     | 0.392   | 0.392    | 3.3                        |
| conv base last 2 layers trainable | 0.999     | 0.922   | 0.969    | 3.2                        |


### Difference

- The first model is fully trainable
  - It fails because it has to much parameters it can train
  - Takes an long time to train
- The second model is the conv base not trainable
  - This gives an better prediction then the base model
  - Relative fast to train
- 4 trainable layers also doesn't learn
  - To much parameters again
- 2 trainable layers works best
  - Better then no trainable layers
  - Doesn't take to much time to train


## vgg19
### Summary

| Model                             | Train_acc | Val_acc | Test_acc | Training Time (in minutes) |
| :-------------------------------- | :-------- | :------ | :------- | :------------------------- |
| conv base not trainable           | 1         | 0.922   | 0.957    | 4.5                        |
| conv base last 4 layers trainable | 0.389     | 0.392   | 0.392    | 4.3                        |
| conv base last 2 layers trainable | 1         | 0.957   | 0.988    | 4.3                        |


### Difference

- The model with 2 trainable layers gives very good results
- 4 trainable layers doesn't train
## RESNET
### Summary

| Model                             | Train_acc | Val_acc | Test_acc | Training Time (in minutes) |
| :-------------------------------- | :-------- | :------ | :------- | :------------------------- |
| ResNet101 not trainable           | 0.997     | 0.937   | 0.969    | 3.6                        |
| ResNet101 last 4 layers trainable | 1         | 0.91    | 0.949    | 3.5                        |
| ResNet101 last 2 layers trainable | 0.997     | 0.933   | 0.961    | 3.6                        |
| ResNet152 last 4 layers trainable | 0.998     | 0.933   | 0.973    | 5.3                        |
| ResNet152 last 2 layers trainable | 1         | 0.937   | 0.957    | 5.4                        |
| ResNet50 last 4 layers trainable  | 0.997     | 0.933   | 0.969    | 2.6                        |
| ResNet50 last 2 layers trainable  | 0.998     | 0.914   | 0.949    | 2.7                        |

### Difference
- ResNet101 has very nice results
  - no trainable works best
  - but 2 trainable layers is close (compared on val accuracy and test accuracy)
- ResNet152 also works very well
  - 4 trainable layers is the best
- ResNet152 also works very well
  - 4 trainable layers is the best
- ResNet152 4 trainable layers is the best model over validation and test accuracy

## RESNET_v2
### Summary

| Model                               | Train_acc | Val_acc | Test_acc | Training Time (in minutes) |
| :---------------------------------- | :-------- | :------ | :------- | :------------------------- |
| ResNet152V2 not trainable           | 0.998     | 0.937   | 0.969    | 5.1                        |
| ResNet152V2 last 4 layers trainable | 0.998     | 0.922   | 0.949    | 4.6                        |
| ResNet152V2 last 2 layers trainable | 0.998     | 0.914   | 0.949    | 5.5                        |


### Difference
- This also has nice results but ResNet152 v1 performes betteru
## xception
### Summary

| Model                            | Train_acc | Val_acc | Test_acc | Training Time (in minutes) |
| :------------------------------- | :-------- | :------ | :------- | :------------------------- |
| xception last 4 layers trainable | 0.999     | 0.882   | 0.941    | 2.5                        |
| xception last 2 layers trainable | 0.999     | 0.894   | 0.949    | 2.4                        |

### Difference

- Here the model with 2 trainable layers performes better