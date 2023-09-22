# Covid-19 Detection Using Chest X-Ray Images

Building a deep learning model to predict COVID-19 positive or negative cases using ResNet-50 involves several steps, including data preprocessing, model architecture, training, and evaluation. Here's an overview of the steps involved:

### Data Collection and Preparation:

Collect a dataset of chest X-ray images labeled as COVID-19 positive and negative.
Split the dataset into training, validation, and test sets to assess the model's performance properly.
### Data Preprocessing:

Resize the images to a consistent size suitable for ResNet-50 (e.g., 224x224 pixels).
Normalize the pixel values to a range suitable for the model (e.g., rescale the pixel values to [0, 1]).
Augment the training dataset using techniques like random rotations, flips, and brightness adjustments to increase the data diversity and help the model generalize better.
### Model Architecture:

Load the pre-trained ResNet-50 model, which includes weights learned from a large dataset (e.g., ImageNet).
Optionally, freeze some initial layers (transfer learning) to retain the pre-trained knowledge and only fine-tune the later layers to adapt to the new task.
Add a few additional layers on top of ResNet-50 to tailor the model for the specific classification task.
Add a final Dense layer with two units and a softmax activation function to obtain the probability distribution over the two classes (COVID-19 positive and negative).

### Model Compilation:

Specify the loss function suitable for binary classification, such as binary cross-entropy.
Choose an appropriate optimizer (e.g., Adam) to update the model's weights during training.
Select evaluation metrics, such as accuracy, to monitor the model's performance during training.

### Model Training:

Train the model using the training dataset, ensuring to use the augmented data.
Set the number of epochs (iterations over the entire training dataset) and the batch size (number of samples processed together).
Monitor the model's performance on the validation set and use early stopping to prevent overfitting.

### Model Evaluation:

After Evaluating got a validation accuracy of 97.51% and a test accuracy of 96.51%


### Fine-tuning and Hyperparameter Tuning:
Experiment with different hyperparameters like learning rate, dropout rate, and optimizer settings to optimize model performance.
Fine-tune the model architecture and hyperparameters based on validation performance.
