import os
from midi2audio import FluidSynth
import random

midi_folder = "/user-fs/chenzihao/dingli/XMIDI_Dataset"
wav_folder = "/user-fs/chenzihao/dingli/XMIDI_output_wav"
sampled_wav_folder = "/user-fs/chenzihao/dingli/onlymusic_test_10000"
soundfont_path = "/user-fs/chenzihao/dingli/third_party/Splendid_136.sf2" 

midi_files = [f for f in os.listdir(midi_folder) if f.endswith('.midi')]
wav_files = [f for f in os.listdir(wav_folder) if f.endswith('.wav')]

unconverted_midi_files = [f for f in midi_files if f.replace('.midi', '.wav') not in wav_files]
sampled_midi_files = random.sample(unconverted_midi_files, min(10000, len(unconverted_midi_files)))

fs = FluidSynth(soundfont_path)
if not os.path.exists(sampled_wav_folder):
    os.makedirs(sampled_wav_folder)
for midi_file in sampled_midi_files:
    midi_path = os.path.join(midi_folder, midi_file)
    wav_file = midi_file.replace('.midi', '.wav')
    wav_path = os.path.join(sampled_wav_folder, wav_file)
    
    fs.midi_to_audio(midi_path, wav_path)
    print(f"Successfully converted {midi_file} to WAV.")

print(f"Successfully converted {len(sampled_midi_files)} .midi files to .wav files and saved them to {sampled_wav_folder}")
