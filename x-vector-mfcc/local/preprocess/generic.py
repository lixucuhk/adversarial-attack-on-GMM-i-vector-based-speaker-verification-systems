import librosa
import librosa.filters
import numpy as np
from scipy import signal
from scipy.io import wavfile
import soundfile as sf
import kaldi_io

def load_wav_snf(path):
    wav, sr = sf.read(path, dtype=np.float32)
    return wav

def load_wav(path, sr=16000):
    return librosa.core.load(path, sr=sr)[0]

def save_wav_snf(wav, path, sr=16000):
    sf.write(path, wav, sr)
    
def save_wav(wav, path, sr=16000):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    # librosa.output.write_wav(path, wav.astype(np.int16), sr)
    wavfile.write(path, sr, wav.astype(np.int16))

def preemphasis(wav, k=0.97):
    return signal.lfilter([1, -k], [1], wav)

def inv_preemphasis(wav, k=0.97):
    return signal.lfilter([1], [1, -k], wav)


def stft(wav, n_fft=512, hop_length=160, win_length=400, window="hann"):
    """Compute Short-time Fourier transform (STFT).
    Returns:
        D:np.ndarray [shape=(t, 1 + n_fft/2), dtype=dtype]
        STFT matrix
    """
    tmp = librosa.stft(y=wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)  # [1 + n_fft/2, t]
    return tmp.T  # (t, 1 + n_fft/2)

def inv_magphase(mag, phase_angle):
    phase = np.cos(phase_angle) + 1.j * np.sin(phase_angle)
    return mag * phase

def power_db_to_mag(power_db):
    power_spec = librosa.core.db_to_power(S_db=power_db, ref=1.0)
    mag_spec = np.sqrt(power_spec) 
    return mag_spec

# def revert_power_db_to_wav(flac_file, power_db, n_fft=1724, hop_length=130, win_length=1724, window="blackman"):
#     wav = load_wav_snf(flac_file)
#     spec = stft(wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
#     _, phase = librosa.magphase(spec)
#     mag = power_db_to_mag(power_db).T
#     complex_specgram = mag * phase
#     audio = librosa.istft(complex_specgram, hop_length=hop_length, win_length=win_length, window=window)
#     return audio

def revert_power_db_to_wav(gt_spec, adv_power_db, n_fft=1724, hop_length=130, win_length=1724, window="blackman"):
    # wav = load_wav_snf(flac_file)
    # spec = stft(wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
    _, phase = librosa.magphase(gt_spec.T)
    phase = phase[:, :adv_power_db.shape[0]]
    mag = power_db_to_mag(adv_power_db).T
    complex_specgram = mag * phase
    audio = librosa.istft(complex_specgram, hop_length=hop_length, win_length=win_length, window=window)
    return audio

def extract_adv_mat(spoofed_feats, vads, ori_feats):
    spoofed_data = { key:mat for key, mat in kaldi_io.read_mat_scp(spoofed_feats)}
    vad_data = { key:vec for key, vec in kaldi_io.read_vec_flt_scp(vads)}
    ori_data = { key:mat for key, mat in kaldi_io.read_mat_scp(ori_feats)}

    num_spoofed = len(spoofed_data.keys())
    num_vad = len(vad_data.keys())
    num_ori = len(ori_data.keys())

    trial_keys = list(spoofed_data.keys())

    assert num_vad == num_ori, \
           "Length does not match! (%d %d)" %(num_vad, num_ori)

    gen_mat = []
    for key in trial_keys:
        print('Process %s utts.' %(key))
        spoofed_mat = spoofed_data.get(key)
        testkey = key[26:]
        vad_vec = vad_data.get(testkey)
        ori_mat = ori_data.get(testkey)
        assert vad_vec is not None, 'No vad for %s %s' %(key, testkey)
        assert ori_mat is not None, 'No original feats for %s %s' %(key, testkey)
        sen_mat = []
        k = 0
        for j in range(len(vad_vec)):
            if vad_vec[j] == 1.0:
                sen_mat.append(spoofed_mat[k])
                k = k+1
            else:
                sen_mat.append(ori_mat[j])

        sen_mat = np.stack(sen_mat, 0)
        gen_mat.append(sen_mat)

    return trial_keys, gen_mat

def extract_adv_mat_frm_grads(grads, vads, ori_feats, sigma):
    grads_data = { key:mat for key, mat in kaldi_io.read_mat_scp(grads)}
    vad_data = { key:vec for key, vec in kaldi_io.read_vec_flt_scp(vads)}
    ori_data = { key:mat for key, mat in kaldi_io.read_mat_scp(ori_feats)}

    num_spoofed = len(grads_data.keys())
    num_vad = len(vad_data.keys())
    num_ori = len(ori_data.keys())

    trial_keys = list(grads_data.keys())

    assert num_vad == num_ori, \
           "Length does not match! (%d %d)" %(num_vad, num_ori)

    gen_mat = []
    for key in trial_keys:
        print('Process %s utts.' %(key))
        grads_mat = grads_data.get(key)
        testkey = key[26:]
        vad_vec = vad_data.get(testkey)
        ori_mat = ori_data.get(testkey)
        assert vad_vec is not None, 'No vad for %s %s' %(key, testkey)
        assert ori_mat is not None, 'No original feats for %s %s' %(key, testkey)
        sen_mat = []
        k = 0
        for j in range(len(vad_vec)):
            if vad_vec[j] == 1.0:
                sen_mat.append(grads_mat[k]*sigma+ori_mat[j])
                k = k+1
            else:
                sen_mat.append(ori_mat[j])

        sen_mat = np.stack(sen_mat, 0)
        gen_mat.append(sen_mat)

    return trial_keys, gen_mat


def extract_adv_voiced_feats(grads, vads, ori_feats, sigma):
    grads_data = { key:mat for key, mat in kaldi_io.read_mat_scp(grads)}
    vad_data = { key:vec for key, vec in kaldi_io.read_vec_flt_scp(vads)}
    ori_data = { key:mat for key, mat in kaldi_io.read_mat_scp(ori_feats)}

    num_spoofed = len(grads_data.keys())
    num_vad = len(vad_data.keys())
    num_ori = len(ori_data.keys())

    trial_keys = list(grads_data.keys())

    assert num_vad == num_ori, \
           "Length does not match! (%d %d)" %(num_vad, num_ori)

    gen_mat = []
    for key in trial_keys:
        print('Process %s utts.' %(key))
        grads_mat = grads_data.get(key)
        testkey = key[26:]
        vad_vec = vad_data.get(testkey)
        ori_mat = ori_data.get(testkey)
        assert vad_vec is not None, 'No vad for %s %s' %(key, testkey)
        assert ori_mat is not None, 'No original feats for %s %s' %(key, testkey)
        sen_mat = []
        k = 0
        for j in range(len(vad_vec)):
            if vad_vec[j] == 1.0:
                sen_mat.append(grads_mat[k]*sigma+ori_mat[j])
                k = k+1

        sen_mat = np.stack(sen_mat, 0)
        gen_mat.append(sen_mat)

    return trial_keys, gen_mat


def uttid2wav(wavfile):
    uttid2wav = {}
    with open(wavfile, 'r') as f:
        for line in f.readlines():
            uttid, wav = line.split()
            uttid2wav.update({uttid:wav})

    return uttid2wav


if __name__ == '__main__':
    spoofed_feats = "data/voxceleb1_test/spoofed_voiced_feats_sample.scp"
    ori_feats = "data/voxceleb1_test/feats_3.scp"
    vad = "data/voxceleb1_test/vad_3.scp"

    keys, mat_list = extract_adv_mat(spoofed_feats, vad, ori_feats)
    wfile = open('./test.txt', 'w')
    wfile.write(keys[0]+'  [\n')
    mat = mat_list[0]
    for vec in mat:
        wfile.write('  ')
        for value in vec:
            wfile.write('%f ' %(value))
        wfile.write('\n')

    wfile.close()
