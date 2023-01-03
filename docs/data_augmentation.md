# Data augmentation
## Summary

| Model                                                       | Train_acc | Val_acc | Test_acc | Training Time (in minutes) |
| :---------------------------------------------------------- | :-------- | :------ | :------- | :------------------------- |
| Baseline                                                    | 1         | 0.69    | 0.69     | 1.9                        |
| Rotation_and_zoom (0.1, 0.2)                                | 0.633     | 0.604   | 0.631    | 3.6                        |
| Rotation_and_zoom (0.9, 0.9)                                | 0.516     | 0.533   | 0.592    | 3.9                        |
| Rotation_and_zoom (0.5, 0.5)                                | 0.548     | 0.533   | 0.643    | 4.3                        |
| Flip, rotation_and_zoom (horizontal, 0.1, 0.2)              | 0.679     | 0.608   | 0.671    | 4.1                        |
| Flip, rotation_and_zoom (vertical, 0.1, 0.2)                | 0.677     | 0.588   | 0.592    | 3.8                        |
| Flip, rotation_and_zoom (horizontal_and_vertical, 0.1, 0.2) | 0.643     | 0.482   | 0.537    | 3.9                        |


## Baseline
- 50 epoches were used to give the model a better chance to against data augmentation
- Some changes were made to the environment so the times will be a lot different between this experiment and the previous experiments.
  - These changes can be found in  [problem solving](problem_solving.md#conda-envs)


## Rotation and zoom
- Rotation and zoom were added to the model.
- The training accuracy is a lot lower then the baseline but the Val and test accuracy is in the same line as the train
  - So definitely no overfitting
  - Also the model not good enough
    - Other changes need to be tested with data augmentation active
    - So the improvements will be better visible

## Flip, rotation and zoom
- Flip, rotation and zoom were added to the model.
- Again there were no improvements here compaired to the baseline.
  - But Flip, rotation_and_zoom (horizontal, 0.1, 0.2) is the best data augmentation so this will be used going further


