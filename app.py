import os
import sys
import tensorflow as tf
import scipy
from scipy import interpolate
from scipy.signal import decimate
import numpy as np
import librosa
import soundfile as sf 
from time import time
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__, )
CORS(app)

SERVER_URL = "http://35.226.138.244:5000"
TMP_DIR = 'static'
	
ckpt = "model/model.ckpt-2077"
	
layers = 2								
n_dim = 0
r = 4
pool_size = 8
strides  = 8
sess = tf.Session()
									
if os.path.isdir(ckpt): 
  checkpoint = tf.train.latest_checkpoint(ckpt)
else: 
  checkpoint = ckpt

meta = checkpoint + '.meta'
		
saver = tf.train.import_meta_graph(meta)
g = tf.get_default_graph()

saver.restore(sess, checkpoint)

X_in, Y_in, alpha_in = tf.get_collection('inputs')
predictions = tf.get_collection('preds')[0]

k_tensors = [n for n in g.as_graph_def().node if 'keras_learning_phase' in n.name]
if k_tensors: 
	k_learning_phase = g.get_tensor_by_name(k_tensors[0].name + ':0')
else:
	print('No keras_learning_phase node')
	exit(0)
		
def spline_up(x_lr, r):
  x_lr = x_lr.flatten()
  x_hr_len = len(x_lr) * r
  x_sp = np.zeros(x_hr_len)
  
  i_lr = np.arange(x_hr_len, step=r)
  i_hr = np.arange(x_hr_len)
  
  f = interpolate.splrep(i_lr, x_lr)

  x_sp = interpolate.splev(i_hr, f)

  return x_sp
  
def get_feed_dict(X):	
	feed_dict = {X_in : X, alpha_in : 1, k_learning_phase: False}
	g = tf.get_default_graph()	
	return feed_dict
		
def predict(X):
    X = X[:len(X) - (len(X) % (2**(layers+1)))]
    X = X.reshape((1,len(X),1))
    feed_dict = get_feed_dict(X)
    return sess.run(predictions, feed_dict=feed_dict)

def upsample_wav(wav_file):
	patch_size = 8192
	x_lr, _ = librosa.load(wav_file, sr=8000)  
	x_lr = np.pad(x_lr, (0, patch_size - (x_lr.shape[0] % patch_size)), 'constant', constant_values=(0,0))
	P = predict(spline_up(x_lr, 2))
	x_hr = P.flatten()
	return x_lr, x_hr
	
@app.route('/')
def index():
	return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
	file = request.files.get('file')
	filename = str(time())
	filepath = os.path.join(TMP_DIR, filename + ".wav")
	file.save(filepath)
	
	x_lr, x_hr = upsample_wav(filepath)
	sf.write(os.path.join(TMP_DIR, filename + '.lr.wav'), x_lr, 8000)
	sf.write(os.path.join(TMP_DIR, filename + '.hr.wav'), x_hr, 16000)
	
	return jsonify({'low_url': f'{SERVER_URL}/static/{filename}.lr.wav', 'high_url': f'{SERVER_URL}/static/{filename}.hr.wav'})

if __name__ == '__main__':
	app.run(host='0.0.0.0')
	
