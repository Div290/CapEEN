# CapEEN: Image Captioning with Early Exits and Knowledge Distillation

## This is the official repository for the paper CapEEN
The entire procedure could be completed in three steps:

1. Initially, the requirements could be installed using the requirements.txt file

>pip install -r requirements.txt

2. After this step, run the train.py file and the backbone will be fine-tuned; save the best model as a checkpoint.

3. Then run the file train_exits.py, which attaches exits to the backbone and learns the weights.
 Note that this step requires the best model from Step 2. In this step, early classifiers are trained and the model is ready to be tested.

The learned model could be then tested on the test.py file and the results could be checked.

Furthermore, the file Acapeen.py could be used to adaptively learn the thresholds for the exits based on the noise present in an image.
