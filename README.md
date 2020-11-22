# Explainable COVID-19 Pneumonia Project CS677 Fall 2020

## Authors
David Apolinar	Da468@njit.edu
Shawn Cicoria	sc2443@njit.edu
Ted Moore	tm437@njit.edu


# Source
This repo is nearly identical to the repo at https://github.com/aildnont/covid-cxr.

There are essentially 2 minor differences - full differences shown below

1. ./config.yml - updated for specific run needs as documented
2. ./src/train.py - here the custom F1 score was dropped as it was not used, nor accurate; in addition couple bugs/issues with respect to python lists vs dictionairies needed for proper processing.


# Setup and running
These steps are the high level steps that are in more detail articulated in the original README here: [README.orig.md Getting Started](README.orig.md#getting-started)

## Getting Started
1. Clone this repository (for help see this
   [tutorial](https://help.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository)).
2. Install the necessary dependencies (listed in
   [requirements.txt](requirements.txt)). To do this, open a terminal in
   the root directory of the project and run the following:
   ```
   $ pip install -r requirements.txt
   ```
3. Create a new folder to contain all of your raw data. Set the _RAW_DATA_
   field in the _PATHS_ section of [config.yml](config.yml) to the
   address of this new folder.
4. Clone the
   [covid-chestxray-dataset](https://github.com/ieee8023/covid-chestxray-dataset)
   repository inside of your _RAW_DATA_ folder. Set the _MILA_DATA_
   field in the _PATHS_ section of [config.yml](config.yml) to the
   address of the root directory of the cloned repository (for help see
   [Project Config](#project-config)).
5. Clone the
   [Figure1-COVID-chestxray-dataset](https://github.com/agchung/Figure1-COVID-chestxray-dataset)
   repository inside of your _RAW_DATA_ folder. Set the _FIGURE1_DATA_
   field in the _PATHS_ section of [config.yml](config.yml) to the
   address of the root directory of the cloned repository.
6. Download and unzip the
   [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)
   dataset from Kaggle somewhere on your local machine. Set the
   _RSNA_DATA_ field in the _PATHS_ section of
   [config.yml](config.yml) to the address of the folder containing the
   dataset.
7. Execute [_preprocess.py_](src/data/preprocess.py) to create Pandas
   DataFrames of filenames and labels. Preprocessed DataFrames and
   corresponding images of the dataset will be saved within
   _data/preprocessed/_.

   ```
   python ./src/data/preprocess.py
   ```
8. Execute [_train.py_](src/train.py) to train the neural network model.
   The trained model weights will be saved within _results/models/_, and
   its filename will resemble the following structure:
   modelyyyymmdd-hhmmss.h5, where yyyymmdd-hhmmss is the current time.
   The [TensorBoard](https://www.tensorflow.org/tensorboard) log files
   will be saved within _results/logs/training/_.
   ```
   python ./src/train.py
   ```
9. In [config.yml](config.yml), set _MODEL_TO_LOAD_ within _PATHS_ to
   the path of the model weights file that was generated in step 6 (for
   help see [Project Config](#project-config)). Execute
   [_lime_explain.py_](src/interpretability/lime_explain.py) to generate
   interpretable explanations for the model's predictions on the test
   set. See more details in the [LIME Section](#lime).
  ```
  python ./src/interpretability/lime_explain.py
  ```

# Preserve and save the model file
As explained above, you need to copy or use the fill lpath to the file for the SHAP steps in the next part of this project.  Again, the model file is in 
_results/models/_.


## File Differences:

### config.yaml

```python
diff --git a/config.yml b/config.yml
index 5af9bbd..d63702b 100644
--- a/config.yml
+++ b/config.yml
@@ -1,8 +1,8 @@
 PATHS:
-  RAW_DATA: 'D:/Documents/Work/covid-cxr/data/'                                       # Path containing all 3 raw datasets (Mila, Figure 1, RSNA)
-  MILA_DATA: 'D:/Documents/Work/covid-cxr/data/covid-chestxray-dataset/'              # Path of Mila dataset https://github.com/ieee8023/covid-chestxray-dataset
-  FIGURE1_DATA: 'D:/Documents/Work/covid-cxr/data/Figure1-COVID-chestxray-dataset/'   # Path of Figure 1 dataset https://github.com/agchung/Figure1-COVID-chestxray-dataset
-  RSNA_DATA: 'D:/Documents/Work/covid-cxr/data/rsna/'                                 # Path of RSNA dataset https://www.kaggle.com/c/rsna-pneumonia-detection-challenge
+  RAW_DATA: 'E:/g/njit/deep-learn/cnn/RAW_DATA/'                                       # Path containing all 3 raw datasets (Mila, Figure 1, RSNA)
+  MILA_DATA: 'E:/g/njit/deep-learn/cnn/RAW_DATA/covid-chestxray-dataset/'              # Path of Mila dataset https://github.com/ieee8023/covid-chestxray-dataset
+  FIGURE1_DATA: 'E:/g/njit/deep-learn/cnn/RAW_DATA/Figure1-COVID-chestxray-dataset/'   # Path of Figure 1 dataset https://github.com/agchung/Figure1-COVID-chestxray-dataset
+  RSNA_DATA: 'E:/g/njit/deep-learn/cnn/RAW_DATA/rsna/'                                 # Path of RSNA dataset https://www.kaggle.com/c/rsna-pneumonia-detection-challenge
   PROCESSED_DATA: 'data/processed/'
   TRAIN_SET: 'data/processed/train_set.csv'
   VAL_SET: 'data/processed/val_set.csv'
@@ -10,10 +10,11 @@ PATHS:
   IMAGES: 'documents/generated_images/'
   LOGS: 'results\\logs\\'
   MODEL_WEIGHTS: 'results/models/'
-  MODEL_TO_LOAD: 'results/models/model.h5'
+  # MODEL_TO_LOAD: 'results/models/model.h5'
+  MODEL_TO_LOAD: 'results\models\model20201110-073719.h5'
   LIME_EXPLAINER: './data/interpretability/lime_explainer.pkl'
   OUTPUT_CLASS_INDICES: './data/interpretability/output_class_indices.pkl'
-  BATCH_PRED_IMGS: 'data/processed/test/'
+  BATCH_PRED_IMGS: 'E:/g/njit/deep-learn/cnn/RAW_DATA/Figure1-COVID-chestxray-dataset/images/'
   BATCH_PREDS: 'results/predictions/'
 
 DATA:
@@ -27,18 +28,21 @@ DATA:
 
 TRAIN:
   CLASS_MODE: 'binary'                                    # One of {'binary', 'multiclass'}
-  MODEL_DEF: 'dcnn_resnet'                                # One of {'dcnn_resnet', 'resnet50v2', 'resnet101v2'}
+  MODEL_DEF: 'resnet50v2'                                # One of {'dcnn_resnet', 'resnet50v2', 'resnet101v2'}
   CLASS_MULTIPLIER: [0.15, 1.0]                           # Class multiplier for binary classification
   #CLASS_MULTIPLIER: [0.4, 1.0, 0.4]                      # Class multiplier for multiclass classification (3 classes)
   EXPERIMENT_TYPE: 'single_train'                         # One of {'single_train', 'multi_train', 'hparam_search'}
   BATCH_SIZE: 32
   EPOCHS: 200
+  # EPOCHS: 100 # SPC
   THRESHOLDS: 0.5                                         # Can be changed to list of values in range [0, 1]
   PATIENCE: 7
-  IMB_STRATEGY: 'class_weight'                            # One of {'class_weight', 'random_oversample'}
+  # IMB_STRATEGY: 'class_weight'                            # One of {'class_weight', 'random_oversample'}
+  IMB_STRATEGY: 'random_oversample'                            # One of {'class_weight', 'random_oversample'} # SPC
   METRIC_PREFERENCE: ['auc', 'recall', 'precision', 'loss']
   NUM_RUNS: 10
   NUM_GPUS: 1
+  VERBOSE: 2 # TODO: SPC
 
 NN:
   DCNN_BINARY:
@@ -72,6 +76,7 @@ LIME:
   NUM_FEATURES: 1000
   NUM_SAMPLES: 1000
   COVID_ONLY: false
 
 HP_SEARCH:
   METRICS: ['accuracy', 'loss', 'recall', 'precision', 'auc']

```

### train.py

```python
diff --git a/src/train.py b/src/train.py
index b54ef4b..c5248db 100644
--- a/src/train.py
+++ b/src/train.py
@@ -16,7 +16,7 @@ from tensorflow.keras.preprocessing.image import ImageDataGenerator
 from tensorboard.plugins.hparams import api as hp
 from src.models.models import *
 from src.visualization.visualize import *
-from src.custom.metrics import F1Score
+# from src.custom.metrics import F1Score
 from src.data.preprocess import remove_text
 
 def get_class_weights(histogram, class_multiplier=None):
@@ -31,7 +31,7 @@ def get_class_weights(histogram, class_multiplier=None):
         weights[i] = (1.0 / len(histogram)) * sum(histogram) / histogram[i]
     class_weight = {i: weights[i] for i in range(len(histogram))}
     if class_multiplier is not None:
-        class_weight = [class_weight[i] * class_multiplier[i] for i in range(len(histogram))]
+         class_weight = {i: class_weight[i] * class_multiplier[i] for i in range(len(histogram))}  # YTODO: PaulA
     print("Class weights: ", class_weight)
     return class_weight
 
@@ -105,11 +105,11 @@ def train_model(cfg, data, callbacks, verbose=1):
     # Define metrics.
     covid_class_idx = test_generator.class_indices['COVID-19']   # Get index of COVID-19 class
     thresholds = 1.0 / len(cfg['DATA']['CLASSES'])      # Binary classification threshold for a class
-    metrics = ['accuracy', CategoricalAccuracy(name='accuracy'),
+    metrics = [CategoricalAccuracy(name='accuracy'), # TODO: Paul
                Precision(name='precision', thresholds=thresholds, class_id=covid_class_idx),
                Recall(name='recall', thresholds=thresholds, class_id=covid_class_idx),
-               AUC(name='auc'),
-               F1Score(name='f1score', thresholds=thresholds, class_id=covid_class_idx)]
+               AUC(name='auc')] #,
+               #F1Score(name='f1score', thresholds=thresholds, class_id=covid_class_idx)]
 
     # Define the model.
     print('Training distribution: ', ['Class ' + list(test_generator.class_indices.keys())[i] + ': ' + str(histogram[i]) + '. '
@@ -141,6 +141,7 @@ def train_model(cfg, data, callbacks, verbose=1):
                                   verbose=verbose, class_weight=class_weight)
 
     # Run the model on the test set and print the resulting performance metrics.
+    print(' *** evaluation on test set *** ')
     test_results = model.evaluate_generator(test_generator, verbose=1)
     test_metrics = {}
     test_summary_str = [['**Metric**', '**Value**']]
@@ -177,7 +178,7 @@ def multi_train(cfg, data, callbacks, base_log_dir):
             cur_callbacks.append(TensorBoard(log_dir=log_dir, histogram_freq=1))
 
         # Train the model and evaluate performance on test set
-        new_model, test_metrics, test_generator = train_model(cfg, data, cur_callbacks, verbose=1)
+        new_model, test_metrics, test_generator = train_model(cfg, data, cur_callbacks, verbose=cfg['TRAIN']['VERBOSE'])
 
         # Log test set results and images
         if base_log_dir is not None:
@@ -338,6 +339,16 @@ def train_experiment(cfg=None, experiment='single_train', save_weights=True, wri
     if cfg is None:
         cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))
 
+    # HACK: TODO: 
+    import tensorflow as tf
+    gpus = tf.config.experimental.list_physical_devices('GPU')
+    if gpus:
+        try:
+            for gpu in gpus:
+                tf.config.experimental.set_memory_growth(gpu, True)
+        except RuntimeError as e:
+            print(e)
+
     # Set logs directory
     cur_date = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
     log_dir = cfg['PATHS']['LOGS'] + "training\\" + cur_date if write_logs else None
@@ -376,5 +387,13 @@ def train_experiment(cfg=None, experiment='single_train', save_weights=True, wri
 
 
 if __name__ == '__main__':
+    # config.gpu_options.allow_growth = True
     cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))
+
+    # HACK TODO Remove  results\models\model20201115-123712.h5
+    # HACK https://github.com/tensorflow/tensorflow/issues/33646#issuecomment-566433261 
+#    from tensorflow.keras.models import load_model
+#    custom_objects={'F1Score':F1Score()}
+#    model_temp = load_model(cfg['PATHS']['MODEL_WEIGHTS'] + 'model20201115-123712.h5', custom_objects=custom_objects, compile=False)
+
     train_experiment(cfg=cfg, experiment=cfg['TRAIN']['EXPERIMENT_TYPE'], save_weights=True, write_logs=True)
\ No newline at end of file

```
