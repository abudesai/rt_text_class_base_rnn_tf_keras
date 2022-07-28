
from multiprocessing.dummy import active_children
import numpy as np, pandas as pd
import joblib
import sys
import os, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
warnings.filterwarnings('ignore') 


import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Layer, GlobalMaxPooling1D, \
    LSTM, GRU, Embedding, Bidirectional
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.nn import softmax


MODEL_NAME = "text_class_RNN_tf_keras"

model_params_fname = "model_params.save"
model_wts_fname = "model_wts.save"
history_fname = "history.json"


COST_THRESHOLD = float('inf')

class InfCostStopCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        loss_val = logs.get('loss')
        if(loss_val == COST_THRESHOLD or tf.math.is_nan(loss_val)):
            print("Cost is inf, so stopping training!!")
            self.model.stop_training = True



def get_init_values(shape): 
    dim = np.prod(shape)
    vals = np.random.randn(dim) / np.sqrt(dim)
    return vals.reshape(shape)


class EarlyStoppingAtMinLoss(Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.

  Arguments:
      patience: Number of epochs to wait after min has been hit. After this
      number of no improvement, training stops.
  """
    def __init__(self, monitor, patience=3, min_epochs=50):
        super(EarlyStoppingAtMinLoss, self).__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_epochs = min_epochs
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience and epoch >= self.min_epochs:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                # print("Restoring model weights from the end of the best epoch.")
                # self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))


class Classifier(): 
    
    def __init__(self, rnn_unit, vocab_size, max_seq_len, num_target_classes, 
                 embedding_size, latent_dim, lr):
        '''
        V: vocabulary size
        T: length of sequences
        K: number of target classes       
        D: embedding size
        M: # of neurons in hidden layer  
        '''
        self.rnn_unit = rnn_unit.lower()     
        self.V = vocab_size
        self.T = max_seq_len
        self.K = num_target_classes
        self.D = embedding_size
        self.M = latent_dim
        self.lr = lr
        self.model = self.build_model()  
        self.model.compile(
            loss=SparseCategoricalCrossentropy(from_logits=True),
            optimizer=Adam(learning_rate=self.lr),
            metrics=['accuracy']
        )   
    
    
    def _get_rnn_unit(self): 
        if self.rnn_unit == 'gru':
            return GRU 
        elif self.rnn_unit == 'lstm':
            return LSTM 
        else: 
            raise ValueError(f"RNN unit {self.rnn_unit} is unrecognized. Must be either lstm or gru.") 
        
        
    def build_model(self): 
        
        rnn = self._get_rnn_unit()
        
        i = Input(shape=(self.T))
        
        x = Embedding( self.V + 1, self. D)(i)
        # x = rnn(self.M, return_sequences=True)(x)
        x = Bidirectional(rnn(self.M, return_sequences=True))(x)
        x = GlobalMaxPooling1D()(x)
        x = Dense(10, activation='relu')(x)
        o = Dense(self.K)(x)
        model = Model(i, o)
        return model
    
    
    def fit(self, train_X, train_y, valid_X=None, valid_y=None,
            batch_size=64, epochs=100, verbose=0):        
        
        if valid_X is not None and valid_y is not None:
            early_stop_loss = 'val_loss' 
            validation_data = [valid_X, valid_y]
        else: 
            early_stop_loss = 'loss'
            validation_data = None   
        
        early_stop_callback = EarlyStopping(monitor=early_stop_loss, patience=10, min_delta=1e-4)    
        
        infcost_stop_callback = InfCostStopCallback()
    
        history = self.model.fit(
                x = train_X,
                y = train_y, 
                batch_size = batch_size,
                validation_data=validation_data,
                epochs=epochs,
                verbose=verbose,
                shuffle=True,
                callbacks=[early_stop_callback, infcost_stop_callback]
            )
        return history
    
    
    def predict(self, X, verbose=False): 
        logits = self.model.predict(X, verbose=verbose)
        return softmax(logits).numpy()
    

    def summary(self):
        self.model.summary()
        
    
    def evaluate(self, x_test, y_test): 
        """Evaluate the model and return the loss and metrics"""
        if self.model is not None:
            return self.model.evaluate(x_test, y_test, verbose=0)        


    def save(self, model_path): 
        model_params = {
            "rnn_unit": self.rnn_unit,
            "vocab_size": self.V,
            "max_seq_len": self.T,
            "num_target_classes": self.K,
            "embedding_size": self.D,
            "latent_dim": self.M,
            "lr": self.lr,
        }
        joblib.dump(model_params, os.path.join(model_path, model_params_fname))
        self.model.save_weights(os.path.join(model_path, model_wts_fname))


    @classmethod
    def load(cls, model_path): 
        model_params = joblib.load(os.path.join(model_path, model_params_fname))
        classifier = cls(**model_params)
        classifier.model.load_weights(os.path.join(model_path, model_wts_fname)).expect_partial()
        return classifier


def save_model(model, model_path):    
    model.save(model_path) 
    

def load_model(model_path):     
    try: 
        model = Classifier.load(model_path)
    except: 
        raise Exception(f'''Error loading the trained {MODEL_NAME} model. 
            Do you have the right trained model in path: {model_path}?''')
    return model


def save_training_history(history, f_path): 
    hist_df = pd.DataFrame(history.history) 
    hist_json_file = os.path.join(f_path, history_fname)
    with open(hist_json_file, mode='w') as f:
        hist_df.to_json(f)



def get_data_based_model_params(train_X, train_y, valid_X, valid_y ): 
    ''' 
        Set any model parameters that are data dependent. 
        For example, number of layers or neurons in a neural network as a function of data shape.
    '''  
    V = max(train_X.max(), valid_X.max()) + 1
    T = train_X.shape[1]
    K = len(set(train_y).union(set(valid_y)))
    return {"vocab_size": V, "max_seq_len": T, "num_target_classes": K}