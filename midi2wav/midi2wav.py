import os
from midi2audio import FluidSynth

midi_dir = "/ckptstorage/chenzihao/dl/xmidi_dataset_10" 
output_dir = "/ckptstorage/chenzihao/dl/xmidi_output_10"  
soundfont_path = "/user-fs/chenzihao/dingli/third_party/Splendid_136.sf2" 

fs = FluidSynth(soundfont_path, 48000)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

converted_count = 0
max_covert = 10

for filename in os.listdir(midi_dir):
    if filename.endswith(".midi") or filename.endswith(".mid"): 
        midi_file_path = os.path.join(midi_dir, filename)
        output_file_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.wav")
        
        fs.midi_to_audio(midi_file_path, output_file_path)
        print(f"Successfully converted {filename} to WAV.")
        
        converted_count += 1
        if converted_count >= max_covert:
            break
