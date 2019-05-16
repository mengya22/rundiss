#-*- coding: utf-8 -*-
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import ipdb
import argparse
import time
import cv2
from tensorflow.contrib import rnn
from utils import sparse_tuple_from as sparse_tuple_from
from utils import pad_sequences as pad_sequences
#from tensorflow.models.rnn import rnn_cell
from keras.preprocessing import sequence

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
config = tf.ConfigProto()  
config.log_device_placement=True
#config.gpu_options.per_process_gpu_memory_fraction=0.4
config.gpu_options.allow_growth = True 

#config.gpu_options.per_process_gpu_memory_fraction = 0.8  # 程序最多只能占用指定gpu50%的显存 

############### Global Parameters ###############
video_data_path= '/data/lianyang/SLR_VideoCaption_cn2_im.csv'
video_feat_path = '/data/guodan/tf.s2vt/slr_sents/slr_feats/slr_C3D_feats'
#video_feat_path = '/media/lmc/guodan/caffe.s2vt/slr/SLR_VGG_DATA/slr_vgg_feats/vgg_feats_key_100'
model_path = 'model/'
test_model='model/model-835'
output_path='captions/model835.txt'
############## Train Parameters #################
dim_image = 4096
dim_hidden= 1000

len_sent=10
dim_embed=256
n_frame_step = 66 #25
n_truncation_step = 27

n_epochs = 1000 #1000
batch_size = 100
learning_rate = 0.001
num_layers=1
SPACE_TOKEN = '<space>'
SPACE_INDEX = 0
num_classes=180
initial_learning_rate = 1e-2
#################################################



##################################################
def fake_data(num_examples, num_features, num_labels, min_size = 10, max_size=15):

    # Generating different timesteps for each fake data
    timesteps = np.random.randint(min_size, max_size, (num_examples,))

    # Generating random input
    inputs = np.asarray([np.random.randn(t, num_features).astype(np.float32) for t in timesteps])

    # Generating random label, the size must be less or equal than timestep in order to achieve the end of the lattice in max timestep
    labels = np.asarray([np.random.randint(0, num_labels, np.random.randint(1, inputs[i].shape[0], (1,))).astype(np.int64) for i, _ in enumerate(timesteps)])

    return inputs, labels
def get_video_data(video_data_path, video_feat_path, data_type='train'):
    video_data = pd.read_csv(video_data_path, sep=',')
    video_data = video_data[video_data['Type'] == data_type]
    #ipdb.set_trace()
    video_data['video_path'] = video_data['VideoID'].map(lambda x: os.path.join(video_feat_path, x +'.npy'))
    video_data = video_data[video_data['video_path'].map(lambda x: os.path.exists( x ))]
    video_data = video_data[video_data['Description'].map(lambda x: isinstance(x, str))]

    return video_data

def preProBuildWordVocab(sentence_iterator, word_count_threshold=1): # borrowed this function from NeuralTalk
    print 'preprocessing word counts and creating vocab based on word count threshold %d' % (word_count_threshold, )
    word_counts = {}
    nsents = 0

    for sent in sentence_iterator:
        nsents += 1
        for w in sent.lower().split(' '):
           word_counts[w] = word_counts.get(w, 0) + 1

    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print 'filtered words from %d to %d' % (len(word_counts), len(vocab))

    ixtoword = {}
    ixtoword[0] = '.'  # period at the end of the sentence. make first dimension be end token
    wordtoix = {}
    wordtoix['#START#'] = 0 # make first vector be the start token
    ix = 1
    for w in vocab:
        wordtoix[w] = ix
        ixtoword[ix] = w
        ix += 1

    word_counts['.'] = nsents
    bias_init_vector = np.array([1.0*word_counts[ixtoword[i]] for i in ixtoword])
    bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range
    return wordtoix, ixtoword, bias_init_vector

def train():
    graph=tf.Graph()
    with graph.as_default():
        inputs = tf.placeholder(tf.float32, [None, None, dim_image])
        targets = tf.sparse_placeholder(tf.int32)
        seq_len = tf.placeholder(tf.int32, [None])

    # Defining the cell
    # Can be:
    #   tf.nn.rnn_cell.RNNCell
    #   tf.nn.rnn_cell.GRUCell
        cell = tf.contrib.rnn.LSTMCell(dim_hidden, state_is_tuple=True)

    # Stacking rnn cells
        stack = tf.contrib.rnn.MultiRNNCell([cell] * num_layers,
                                        state_is_tuple=True)

        outputs, _ = tf.nn.dynamic_rnn(stack, inputs, seq_len, dtype=tf.float32)
        shape = tf.shape(inputs)
        batch_s, max_timesteps = shape[0], shape[1]
        outputs = tf.reshape(outputs, [-1, dim_hidden])

        W = tf.Variable(tf.truncated_normal([dim_hidden,
                                         num_classes],
                                        stddev=0.1))

        b = tf.Variable(tf.constant(0., shape=[num_classes]))
        logits = tf.matmul(outputs, W) + b
        logits = tf.reshape(logits, [batch_s, -1, num_classes])
        logits = tf.transpose(logits, (1, 0, 2))
        loss = tf.nn.ctc_loss(targets, logits, seq_len)
        cost = tf.reduce_mean(loss)
        optimizer = tf.train.MomentumOptimizer(initial_learning_rate,
                                               0.9).minimize(cost)
    #ipdb.set_trace()
    # Option 2: tf.nn.ctc_beam_search_decoder
    # (it's slower but you'll get better results)
        decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)

    # Inaccuracy: label error rate
        ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),
                                          targets))

    with tf.Session(graph=graph) as session:
        train_data = get_video_data(video_data_path, video_feat_path, data_type='train')
        test_data = get_video_data(video_data_path, video_feat_path, data_type='test')
    #ipdb.set_trace()
        captions = train_data['Description'].values
        captions = map(lambda x: x.replace('.', ''), captions)
        captions = map(lambda x: x.replace(',', ''), captions)#(1,4000)zi mu
        wordtoix, ixtoword, bias_init_vector = preProBuildWordVocab(captions, word_count_threshold=0)
        np.save('ixtoword4000', ixtoword)
        saver = tf.train.Saver(max_to_keep=500)
        tf.global_variables_initializer().run()
        for epoch in range(n_epochs+1):
            train_cost=train_ler=0
            start=time.time()
            index = list(train_data.index)
            np.random.shuffle(index)
            train_data = train_data.ix[index]
            current_train_data = train_data.groupby('video_path').apply(lambda x: x.irow(np.random.choice(len(x))))
            current_train_data = current_train_data.reset_index(drop=True)

            for start,end in zip(
                    range(0, len(current_train_data), batch_size),
                    range(batch_size, len(current_train_data), batch_size)):

                current_batch = current_train_data[start:end]#(100,4)
                current_videos = current_batch['video_path'].values

                current_feats = np.zeros((batch_size, n_frame_step, dim_image))
                current_feats_vals = map(lambda vid: np.load(vid), current_videos)
                batch_train_inputs=np.array(current_feats_vals)
                batch_train_inputs,batch_train_seq_len=pad_sequences(batch_train_inputs)
                current_captions = current_batch['Description'].values
                current_caption_ind = map(lambda cap: [wordtoix[word] for word in cap.lower().split(' ') if word in wordtoix], current_captions)
                current_caption_ind=np.array(current_caption_ind) 
                for i in range(100):
                       current_caption_ind[i]=np.array(current_caption_ind[i])
                #inputs_t, labels = fake_data(100, 13, num_classes - 1)
                #batch_train_targets=sparse_tuple_from(labels)
                batch_train_targets=sparse_tuple_from(current_caption_ind)
                feed = {inputs: batch_train_inputs,
                        targets: batch_train_targets,
                        seq_len: batch_train_seq_len}

                batch_cost, _ = session.run([cost, optimizer], feed)
                train_cost += batch_cost*batch_size
                train_ler += session.run(ler, feed_dict=feed)*batch_size    
          
            train_cost /= 4000
            train_ler /= 4000

            log = "Epoch {}/{}, train_cost = {:.3f}, train_ler = {:.3f}, time = {:.3f}"
            print(log.format(epoch+1, n_epochs, train_cost, train_ler, time.time() - start))
           
            if np.mod(epoch, 5) == 0:
                 print "Epoch ", epoch, " is done. Saving the model ..."
                 saver.save(session, os.path.join(model_path, 'model'), global_step=epoch)
    session.close()

def test(model_path=test_model,video_feat_path=video_feat_path):
    graph=tf.Graph()
    with graph.as_default():
        inputs = tf.placeholder(tf.float32, [None, None, dim_image])
        targets = tf.sparse_placeholder(tf.int32)
        seq_len = tf.placeholder(tf.int32, [None])

    # Defining the cell
    # Can be:
    #   tf.nn.rnn_cell.RNNCell
    #   tf.nn.rnn_cell.GRUCell
        cell = tf.contrib.rnn.LSTMCell(dim_hidden, state_is_tuple=True)

    # Stacking rnn cells
        stack = tf.contrib.rnn.MultiRNNCell([cell] * num_layers,
                                        state_is_tuple=True)

        outputs, _ = tf.nn.dynamic_rnn(stack, inputs, seq_len, dtype=tf.float32)
        shape = tf.shape(inputs)
        batch_s, max_timesteps = shape[0], shape[1]
        outputs = tf.reshape(outputs, [-1, dim_hidden])

        W = tf.Variable(tf.truncated_normal([dim_hidden,
                                         num_classes],
                                        stddev=0.1))

        b = tf.Variable(tf.constant(0., shape=[num_classes]))
        logits = tf.matmul(outputs, W) + b
        logits = tf.reshape(logits, [batch_s, -1, num_classes])
        logits = tf.transpose(logits, (1, 0, 2))
        loss = tf.nn.ctc_loss(targets, logits, seq_len)
        cost = tf.reduce_mean(loss)
        optimizer = tf.train.MomentumOptimizer(initial_learning_rate,
                                               0.9).minimize(cost)
    #ipdb.set_trace()
    # Option 2: tf.nn.ctc_beam_search_decoder
    # (it's slower but you'll get better results)
        decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)

    # Inaccuracy: label error rate
        ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),
                                          targets))
    

    with tf.Session(graph=graph) as session:
        
        
        test_data = get_video_data(video_data_path, video_feat_path, data_type='test')
        test_videos=test_data['video_path'].unique()
        ixtoword = pd.Series(np.load('ixtoword4000.npy').tolist())
        saver=tf.train.Saver()
        saver.restore(session,model_path)
        f=open(output_path,'w')
        for video_feat_path in test_videos:
            video_feat=np.load(video_feat_path)[None,...]
            test_input,test_seq_len=pad_sequences(video_feat)
            feed={inputs:test_input,
                  seq_len:test_seq_len
                 }
            d=session.run(decoded[0],feed_dict=feed)
            dense_decoded=tf.sparse_tensor_to_dense(d,default_value=-1).eval(session=session)
            generated_words=ixtoword[dense_decoded.tolist()[0]]
            generated_sentence=' '.join(generated_words)
            print generated_sentence
            f.writelines('%s:%s\n' % (video_feat_path,generated_sentence))
    session.close()
    f.close()
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--type", type=str, required=True,
                    help='train or test')
    args = parser.parse_args()
    if args.type == 'train':
        train()
    else:
        test()

if __name__ == "__main__":
   main()
