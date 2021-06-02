# AutoKeras for image classification
"Image classification using automated machine learning library AutoKeras" BSc work for University of Latvia

Files in the directory:
* oracle.json - contains hyperparameter/architecture space for the model that was stopped (it's almsot the same for all other models that were trained), seeds for reproducibility, initial hyperparameters/architectures.
* best_model_trial.json - contains the best models hyperparameter/architecture space, the values used and training results.
* CIFAR10_ak.py - code that was used to run the AutoKeras training on CIFAR-10 dataset.
* CIFAR10_tutorial.py - code that was used to run the tutorial model (https://www.tensorflow.org/tutorials/images/classification) on CIFAR-10 dataset.

Note: both .py files were converted from jupyter notebook files (.ipynb).