##
import sys
import os

from preprocess.generic import load_wav_snf_scaled, save_wav_snf_scaled

audios_dir = sys.argv[1]
scaled_audios_dir = audios_dir+'_scaled'

os.makedirs(scaled_audios_dir, exist_ok=True)

for file in os.listdir(audios_dir):
	if len(file) < 30:
		continue
	wav, scale_ = load_wav_snf_scaled(audios_dir+'/'+file)
	testfile = file[26:]
	wav_, scale = load_wav_snf_scaled(audios_dir+'/'+testfile)
	save_wav_snf_scaled(wav, scaled_audios_dir+'/'+file, scale)
