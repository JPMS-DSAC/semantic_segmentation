import tensorflow as tf
from tensorflow.python.client import device_lib

print("Device details: ",device_lib.list_local_devices(),end='\n\n\n')
print("Num GPUs: ", len(tf.config.list_physical_devices('GPU')))

from datetime import datetime

#creating extra data from current files

import pandas as pd
import os
from tqdm import tqdm
import numpy as np
from tensorflow import keras
import bert
from bert.tokenization.bert_tokenization import FullTokenizer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
from bert import BertModelLayer

from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras import constraints

from tensorflow.keras import activations
from tensorflow.keras import backend as K

# from keras.utils.np_utils import to_categorical
from tensorflow.keras import optimizers
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import tensorboard

from tensorflow.keras.layers import Layer,Dropout, LSTM, GRU, Bidirectional, TimeDistributed, Dense

from tensorflow.python.client import device_lib

device_lib.list_local_devices()

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

############################### path check (add path to data)
root_path = ''
list_of_files = []
for root, dir, files in os.walk('./Annotated - CSV'):
  root_path = root
  list_of_files = files
  break

list_of_files = ['1314613428609.csv', '1522238936458.csv', '1564575450353.csv', '1409133457223.csv', '1372830044081.csv', '1291175003856_NEG.csv', '1290062946166.csv', '1315463402543.csv', '1324544561749.csv', '1358769139907.csv', '1404444629445.csv', '1427283185104.csv', '1380795608703.csv', '1404800940434.csv', '1382959468059.csv', '1289903641088.csv', '1407404311694.csv', '1494587603795.csv', '1289452697301.csv', '1509190608413.csv', '1372652426612.csv', '1288072330011.csv', '1407404413828.csv', '1404099510806.csv', '1290154724736_NEG.csv', '1407404374671.csv', '1312279794560.csv', '1312280268805.csv', '1288673034598.csv']

print(root_path)
print(list_of_files)

all_dataframes = []
for filename in list_of_files:
  file_path = root_path + '/' + filename
  file_temp = pd.read_csv(file_path)
  all_dataframes.append(file_temp)

for df in all_dataframes:
  context = [sent for sent in df['Sentence']]
  df['context'] = df['Sentence'].apply(lambda x:context)
  df['left'] = df['Sentence'].apply(lambda x:context)
  df['right'] = df['Sentence'].apply(lambda x:context)

result = pd.DataFrame()
result = result.append(all_dataframes,ignore_index=True)

train_data = pd.DataFrame(columns=result.columns)
test_data = pd.DataFrame(columns=result.columns)

print("lul", result.columns)

for label in result.Label.unique():
  temp_df = result[result['Label'] == label]
  train_index = int(temp_df.shape[0]*0.85)
  train_data = train_data.append(temp_df[:train_index])
  test_data = test_data.append(temp_df[train_index:])

# train_data, test_data = train_test_split(result, test_size=0.15)

# train_data.to_csv("train_data.csv")
# test_data.to_csv("test_data.csv")

# train_data = pd.read_csv("train_data.csv", index_col=0, converters={'context': pd.eval, 'left': pd.eval, 'right':pd.eval})
# test_data = pd.read_csv("test_data.csv", index_col=0, converters={'context': pd.eval, 'left': pd.eval, 'right':pd.eval})

context_size = 20

for i, line in train_data.iterrows():
    if context_size==-1:
        break
    ind = line['context'].index(line['Sentence'])
    left = max(ind-context_size, 0)
    right = min(ind+1+context_size, len(line['context']))
    pad_left = 0
    pad_right = 0
    if ind-left<context_size:
        pad_left = context_size-ind+left
        # right+=context_size-(ind-left)
    elif right-ind<context_size:
        pad_right = context_size-ind+right
        # left-=context_size-(right-ind)
    train_data.at[i, 'left'] = line['context'][left:ind]
    train_data.at[i, 'right'] = line['context'][ind+1:right]
    context = line['context'][left:ind] + line['context'][ind+1:right]
    print(left, right, len(context))
    train_data.at[i, 'context'] = context

for i, line in test_data.iterrows():
    if context_size==-1:
        break
    ind = line['context'].index(line['Sentence'])
    left = max(ind-context_size, 0)
    right = min(ind+1+context_size, len(line['context']))
    pad_left = 0
    pad_right = 0
    if ind-left<context_size:
        pad_left = context_size-ind+left
        # right+=context_size-(ind-left)
    elif right-ind<context_size:
        pad_right = context_size-ind+right
        # left-=context_size-(right-ind)
    test_data.at[i, 'left'] = line['context'][left:ind]
    test_data.at[i, 'right'] = line['context'][ind+1:right]
    context = line['context'][left:ind] + line['context'][ind+1:right]
    print(left, right, len(context))
    test_data.at[i, 'context'] = context

train_data.drop(columns = ['Sentence ID'],axis=1,inplace=True)
test_data.drop(columns = ['Sentence ID'],axis=1, inplace=True)

bert_model_name="uncased_L-12_H-768_A-12"
############# path check 
bert_ckpt_dir = os.path.join("model/", bert_model_name)
bert_ckpt_file = os.path.join(bert_ckpt_dir, "bert_model.ckpt")
bert_config_file = os.path.join(bert_ckpt_dir, "bert_config.json")

classes = train_data.Label.unique().tolist()

tokenizer = FullTokenizer(vocab_file=os.path.join(bert_ckpt_dir, "vocab.txt"))

from collections import defaultdict

#preprocessing
class Data_clean:
  #here context referes to document context
  DATA_COLUMN = "Sentence"
  LABEL_COLUMN = "Label"
  LEFT_CONTEXT_COLUMN = "left"
  RIGHT_CONTEXT_COLUMN = "right"
  CONTEXT_COLUMN = "Context"

  def __init__(self, train, test, tokenizer: FullTokenizer, classes, inp_seq_len, max_seq_len):
    self.tokenizer = tokenizer
    self.inp_seq_len = inp_seq_len
    self.max_seq_len = max_seq_len
    self.max_sent_len = 0
    self.classes = classes
    self.dict = defaultdict(int)
    self.sent_set = set()

    train, test = map(lambda df: df.reindex(df[Data_clean.DATA_COLUMN].str.len().sort_values().index), [train, test])

    print(train.count())
    print(test.count())

    ((self.train_x, self.train_x_left_context, self.train_x_right_context, self.train_y), (self.test_x, self.test_x_left_context, self.test_x_right_context, self.test_y)) = map(self._prepare, [train, test])

    print("max seq_len", self.max_seq_len)
    print("max sent_len", self.max_sent_len)
    self.max_seq_len = max(self.max_seq_len, max_seq_len)
    self.train_x, self.test_x = map(self._pad, [self.train_x, self.test_x])
    # self.prepad_train_x_context = self.train_x_context
    # self.prepad_test_x_context = self.test_x_context
    print(self.train_x_left_context.size, self.train_x_right_context.size)
    print(self.test_x_left_context.size, self.test_x_right_context.size)
    self.train_x_left_context, self.test_x_left_context = map(self._context_pad_left, [self.train_x_left_context, self.test_x_left_context])
    print("left context done")
    self.train_x_right_context, self.test_x_right_context = map(self._context_pad_right, [self.train_x_right_context, self.test_x_right_context])
    print("right context done")

  def _prepare(self, df):
    x, y = [], []
    left_context_x = []
    right_context_x = []
    
    for _, row in tqdm(df.iterrows()):
      sent, left_context_text, right_context_text, label = row[Data_clean.DATA_COLUMN], row[Data_clean.LEFT_CONTEXT_COLUMN], row[Data_clean.RIGHT_CONTEXT_COLUMN], row[Data_clean.LABEL_COLUMN]
      sent_tokens = self.tokenizer.tokenize(sent)
      sent_tokens = ["[CLS]"] + sent_tokens[:self.inp_seq_len-2] + ["[SEP]"]
      sent_token_ids = self.tokenizer.convert_tokens_to_ids(sent_tokens)
      self.dict[len(sent_token_ids)]+=1
      # self.max_seq_len = max(self.max_seq_len, len(sent_token_ids))
      x.append(sent_token_ids)
      #context preprocessing
      left_row_context = []
      for c_sent in left_context_text:
        c_sent_tokens = self.tokenizer.tokenize(c_sent)
        c_sent_tokens = ["[CLS]"] + c_sent_tokens[:self.max_seq_len-2] + ["[SEP]"]
        c_sent_token_ids = self.tokenizer.convert_tokens_to_ids(c_sent_tokens)
        
        # self.max_seq_len = max(self.max_seq_len, len(c_sent_token_ids))
        left_row_context.append(c_sent_token_ids)
      
      right_row_context = []
      for c_sent in right_context_text:
        c_sent_tokens = self.tokenizer.tokenize(c_sent)
        c_sent_tokens = ["[CLS]"] + c_sent_tokens[:self.max_seq_len-2] + ["[SEP]"]
        c_sent_token_ids = self.tokenizer.convert_tokens_to_ids(c_sent_tokens)
        
        # self.max_seq_len = max(self.max_seq_len, len(c_sent_token_ids))
        right_row_context.append(c_sent_token_ids)

      self.max_sent_len = max(self.max_sent_len, len(left_row_context))
      self.max_sent_len = max(self.max_sent_len, len(right_row_context))
      left_context_x.append(np.array(left_row_context))
      right_context_x.append(np.array(right_row_context))
      y.append(self.classes.index(label))

    return np.array(x), np.array(left_context_x), np.array(right_context_x), np.array(y)

  def _pad(self, ids):
    x = []
    for input_ids in ids:
      input_ids = input_ids[:min(len(input_ids), self.inp_seq_len - 2)]
      input_ids = input_ids + [0] * (self.inp_seq_len - len(input_ids))
      x.append(np.array(input_ids))
    return np.array(x)

  def _context_pad_right(self, arr):
    context_x = []
    for row in arr:
      row_context = []
      for input_ids in row:
        input_ids = input_ids[:min(len(input_ids), self.max_seq_len - 2)]
        # input_ids = input_ids + [0] * (self.max_seq_len - len(input_ids))
        row_context.append(np.hstack((np.array(input_ids), np.zeros(self.max_seq_len - len(input_ids)))))
      if len(row_context) < self.max_sent_len:
        while len(row_context) < self.max_sent_len:
          padded_sent  = [0] * self.max_seq_len
          row_context.append(padded_sent)
      context_x.append(np.array(row_context))
      # print("lul")
    return np.array(context_x)
  
  def _context_pad_left(self, arr):
    context_x = []
    for row in arr:
      row_context = []
      # print(row.size)
      for input_ids in row:
        input_ids = input_ids[:min(len(input_ids), self.max_seq_len - 2)]
        # input_ids = input_ids + [0] * (self.max_seq_len - len(input_ids))
        row_context.append(np.hstack((np.array(input_ids), np.zeros(self.max_seq_len - len(input_ids)))))
      # print(len(row_context))
      pad = []
      if len(row_context) + len(pad) < self.max_sent_len:
        while len(row_context) + len(pad) < self.max_sent_len:
          padded_sent  = [0] * self.max_seq_len
          pad.append(padded_sent)
      # print(len(pad))
      row_context = pad + row_context
      context_x.append(np.array(row_context))
      # print("CHAL JAJAJAJA")
    return np.array(context_x)

data = Data_clean(train_data, test_data, tokenizer, classes, 150, 150)

# physical_devices = tf.config.list_physical_devices('GPU')
# try:
#   tf.config.experimental.set_memory_growth(physical_devices[0], True)
# except:
#   # Invalid device or cannot modify virtual devices once initialized.
#   pass

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

def dot_product(x, kernel):

    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)
    

class AttentionWithContext(Layer):

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'W_regularizer' : self.W_regularizer,
            'u_regularizer' : self.u_regularizer,
            'b_regularizer' : self.b_regularizer,
            'W_constraint' : self.W_constraint,
            'u_constraint' : self.u_constraint,
            'b_constraint' : self.b_constraint
        })
        return config

    def build(self, input_shape):
        print('len of input shape',input_shape)
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight(shape=(input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

# strategy = tf.distribute.OneDeviceStrategy("device:GPU:0")
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

def printpls(x, str):
    print(str, x)

with strategy.scope():  
  seq_len = data.max_seq_len
  inp_seq_len = data.inp_seq_len
  cls_rem_inp_len = data.inp_seq_len - 1
  cls_rem_seq_len = data.max_seq_len - 1
  sent_len = data.max_sent_len
  # sent_len = 5
  bert_dim = 768

  with tf.io.gfile.GFile(bert_config_file, "r") as reader:
    bc = StockBertConfig.from_json_string(reader.read())
    bert_params = map_stock_config_to_params(bc)
    bert_params.adapter_size = None
    bert = BertModelLayer.from_params(bert_params, name="bert")

  input_ids = keras.layers.Input(shape=(inp_seq_len, ), dtype='int32', name="input_ids")
  printpls(input_ids, 'input_ids')

  bert_output = bert(input_ids)
  printpls(bert_output, 'bert_output')

  cls_out = keras.layers.Lambda(lambda seq: seq[:,1:,:])(bert_output)
  printpls(cls_out, 'cls_out')

  cls_out = keras.layers.Dropout(0.5)(cls_out)
  printpls(cls_out, 'cls_out')

  lstm_out = Bidirectional(GRU(768, return_sequences=True,input_shape=(cls_rem_inp_len, bert_dim)))(cls_out)
  printpls(lstm_out, 'lstm_out')

  lstm_att = AttentionWithContext()(lstm_out)
  printpls(lstm_att, 'lstm_att')

  input_context_left_ids = keras.layers.Input(shape=(sent_len,seq_len,), dtype='int32', name="input_context_left_ids")
  printpls(input_context_left_ids, 'input_context_left_ids')

  context_encoder_left = keras.layers.TimeDistributed(bert)(input_context_left_ids)
  printpls(context_encoder_left, 'context_encoder_left')

  context_encoder_cls_out_left =  keras.layers.TimeDistributed(keras.layers.Lambda(lambda seq: seq[:,1:,:]))(context_encoder_left)
  printpls(context_encoder_cls_out_left, 'context_encoder_cls_out_left')

  lstm_sent_left = keras.layers.TimeDistributed(keras.layers.Bidirectional(GRU(bert_dim, return_sequences=True)))(context_encoder_cls_out_left)
  printpls(lstm_sent_left, 'lstm_sent_left')

  lstm_att_sent_left = keras.layers.TimeDistributed(AttentionWithContext())(lstm_sent_left)
  printpls(lstm_att_sent_left, 'lstm_att_sent_left')

  lstm_doc_left = keras.layers.Bidirectional(GRU(bert_dim, return_sequences=True))(lstm_att_sent_left)
  printpls(lstm_doc_left, 'lstm_doc_left')

  lstm_att_doc_left = AttentionWithContext()(lstm_doc_left)
  printpls(lstm_att_doc_left, 'lstm_att_doc_left')

  input_context_right_ids = keras.layers.Input(shape=(sent_len,seq_len,), dtype='int32', name="input_context_right_ids")
  printpls(input_context_right_ids, 'input_context_right_ids')

  context_encoder_right = keras.layers.TimeDistributed(bert)(input_context_right_ids)
  printpls(context_encoder_right, 'context_encoder_right')

  context_encoder_cls_out_right =  keras.layers.TimeDistributed(keras.layers.Lambda(lambda seq: seq[:,1:,:]))(context_encoder_right)
  printpls(context_encoder_cls_out_right, 'context_encoder_cls_out_right')

  lstm_sent_right = keras.layers.TimeDistributed(keras.layers.Bidirectional(GRU(bert_dim, return_sequences=True)))(context_encoder_cls_out_right)
  printpls(lstm_sent_right, 'lstm_sent_right')

  lstm_att_sent_right = keras.layers.TimeDistributed(AttentionWithContext())(lstm_sent_right)
  printpls(lstm_att_sent_right, 'lstm_att_sent_right')

  lstm_doc_right = keras.layers.Bidirectional(GRU(bert_dim, return_sequences=True))(lstm_att_sent_right)
  printpls(lstm_doc_right, 'lstm_doc_right')

  lstm_att_doc_right = AttentionWithContext()(lstm_doc_right)
  printpls(lstm_att_doc_right, 'lstm_att_doc_right')

  cls_out_concat = keras.layers.Concatenate()([lstm_att_doc_left, lstm_att, lstm_att_doc_right])
  printpls(cls_out_concat, 'cls_out_concat')

  logits = keras.layers.Dense(units=3072,activation="tanh")(cls_out_concat)
  printpls(logits, 'logits')

  logits = keras.layers.Dropout(0.5)(logits)
  printpls(logits, 'logits')

  logits = keras.layers.Dense(units=len(classes), activation="softmax")(logits)
  printpls(logits, 'logits')

  # cls_out_concat = lstm_att_doc
  # printpls(cls_out_concat, 'cls_out_concat')

  # logits = keras.layers.Dense(units=1536,activation="tanh")(cls_out_concat)
  # printpls(logits, 'logits')

  # logits = keras.layers.Dropout(0.5)(logits)
  # printpls(logits, 'logits')

  # logits = keras.layers.Dense(units=len(classes), activation="softmax")(logits)
  # printpls(logits, 'logits')

  model = keras.Model(inputs=[input_ids, input_context_left_ids, input_context_right_ids], outputs=logits)

  load_stock_weights(bert, bert_ckpt_file)

  model.compile(
  optimizer=keras.optimizers.Adam(1e-5),
  loss=keras.losses.SparseCategoricalCrossentropy(),
  metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")]
  )

print(model.summary())

from packaging import version
############## path check 
logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")

#print(tensorboard.__version__)

############## ADD save model path 
my_callbacks = [EarlyStopping(patience=2, monitor="val_loss"),  ModelCheckpoint(filepath='model_{epoch:02d}.hdf5', save_best_only=True, save_weights_only = False, monitor='val_loss', mode='auto',save_freq = 'epoch'),keras.callbacks.TensorBoard(log_dir=logdir)]

train_flag = 1

if train_flag:
  history = model.fit(
    x=[data.train_x, data.train_x_left_context, data.train_x_right_context], 
    y= data.train_y,
    validation_split=0.1,
    batch_size=1,
    shuffle=True,
    epochs = 5,
    callbacks = my_callbacks
  )

else:
  model = tf.keras.models.load_model('./model_03.hdf5', custom_objects={"BertModelLayer": BertModelLayer, "keras":tf.keras, "AttentionWithContext": AttentionWithContext})

_, train_acc = model.evaluate([data.train_x, data.train_x_left_context, data.train_x_right_context], data.train_y)
_, test_acc = model.evaluate([data.test_x, data.test_x_left_context, data.test_x_right_context], data.test_y)

print("train acc", train_acc)
print("test acc", test_acc)

train_pred = model.predict([data.train_x, data.train_x_left_context, data.train_x_right_context])
test_pred = model.predict([data.test_x, data.test_x_left_context, data.test_x_right_context])

classification_report(data.test_y, test_pred, target_names=classes)