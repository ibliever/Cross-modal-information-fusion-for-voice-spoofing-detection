import os
import librosa
import functools
import pickle
import shutil
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import io_ops

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

class Speaker():

	def __init__(self, sample_rate = 16000, duration = 6, mono = True, window = 400, stride = 160, fft_length = 512, amp_norm = 0.3, verbose = False, audios_path="./data/audio/train/", spect_path="./data/spect/train/", flac_path ="./data/flac/train/"):
		self.videos_path = "data/videos/"
		self.flac_path = flac_path
		self.audios_path = audios_path
		self.spect_path  = spect_path
		self.sample_rate = sample_rate
		self.duration = duration
		self.mono = mono
		self.window = window
		self.stride = stride
		self.fft_length = fft_length
		self.amp_norm = amp_norm
		self.verbose = verbose

	def find_spec(self, filename):
		print("-------------finding spectrogram for {0}----------------".format(filename))
		with tf.Session(graph=tf.Graph()) as sess:

			holder = tf.placeholder(tf.string, [])
			file = tf.io.read_file(holder)
			decoder = tf.contrib.ffmpeg.decode_audio(file, file_format = "wav", samples_per_second = self.sample_rate, channel_count = 1)
			stft = tf.signal.stft(tf.transpose(a=decoder), frame_length = self.window, frame_step = self.stride, fft_length = self.fft_length, window_fn = tf.signal.hann_window)
			amp = tf.squeeze(tf.abs(stft)) ** self.amp_norm
			phase = tf.squeeze(tf.math.angle(stft))
			stacked = tf.stack([amp, phase], 2)
			stft = sess.run(stacked, feed_dict = {holder : self.audios_path + filename + ".wav"})
			pickle.dump(stft, open(self.spect_path + filename  + ".pkl", "wb"))
			print("============STFT SHAPE IS {0}=============".format(stft.shape))

	def extract_wav(self, filename):

		wavfile = filename + ".wav"

		if (not os.path.isfile(self.spect_path+filename+".pkl")):
			data, _ = librosa.load(self.flac_path + filename + ".flac", sr = self.sample_rate, mono = self.mono, duration = self.duration)
			fac = int(np.ceil((1.0* self.duration * self.sample_rate)/len(data)))
			updated_data = np.tile(data, fac)[0:(self.duration * self.sample_rate)]
			librosa.output.write_wav(self.audios_path +  wavfile, updated_data, self.sample_rate, norm = False)
			self.find_spec(filename)
		else:
			print(filename+".pkl already exists!!!")


if __name__ == "__main__":
	speaker = Speaker();
