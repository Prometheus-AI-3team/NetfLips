import argparse
import json
import torch
import soundfile as sf
import re
import os
from model import CodeHiFiGANModel_spk
from utils import AttrDict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to the model checkpoint (e.g. g_00011000)')
    parser.add_argument('--config', required=True, help='Path to config.json')
    parser.add_argument('--input_file', required=True, help='Path to input .pt file containing code and optional spkr')
    parser.add_argument('--output_folder', default='output/unit2a', help='Output wav folder')
    parser.add_argument('--device', default='cuda', help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Configuration
    print(f"Loading config from {args.config}...")
    with open(args.config) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    
    # 2. Initialize Model
    print("Initializing model...")
    generator = CodeHiFiGANModel_spk(dict(h)).to(device)
    
    # 3. Load Checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    state_dict = torch.load(args.checkpoint, map_location=device)
    
    if 'generator' in state_dict:
        generator.load_state_dict(state_dict['generator'])
    else:
        # Just in case the checkpoint structure is different
        generator.load_state_dict(state_dict)
        
    generator.eval()
    generator.remove_weight_norm()
    
    # 4. Load Input Data
    print(f"Loading input units from {args.input_file}...")
    # Expecting the .pt file format used in training (containing 'code' and optionally 'spkr')
    data = torch.load(args.input_file, map_location='cpu')
    
    if isinstance(data, dict):
        code = data.get('code')
        spkr = data.get('spkr')
    else:
        # Fallback if it's just a code tensor
        code = data
        spkr = None

    if code is None:
        raise ValueError("Could not find 'code' in the input file.")
        
    # Prepare input for model
    # Model expects batch dimension
    if code.dim() == 1:
        code = code.unsqueeze(0)
    
    x = {'code': code.to(device)}
    
    if h.get('multispkr') and spkr is not None:
        if spkr.dim() == 1:
            spkr = spkr.unsqueeze(0)
        x['spkr'] = spkr.to(device)
        print("Using speaker embedding from input file.")
    elif h.get('multispkr'):
        print("Warning: Model expects speaker embedding but none provided in input file. This may cause errors.")
        
    # 5. Run Inference
    print("Generating audio...")
    with torch.no_grad():
        # returns (wav, dedup_code)
        y_g_hat, _ = generator(**x)
        
    audio = y_g_hat.squeeze()
    audio = audio.cpu().numpy()
    
    # 6. Save Output
    os.makedirs(args.output_folder, exist_ok=True)
    
    input_base = os.path.splitext(os.path.basename(args.input_file))[0][:-13]
    
    # Extract step from checkpoint filename
    ckpt_name = os.path.basename(args.checkpoint)
    match = re.search(r'(\d+)', ckpt_name)
    if match:
        step = int(match.group(1))
        suffix = f"{step}step"
    else:
        suffix = "unknown_step"

    output_filename = f"{input_base}_{suffix}.wav"
    output_path = os.path.join(args.output_folder, output_filename)
    
    print(f"Saving audio to {output_path}...")
    sf.write(output_path, audio, h.sampling_rate)
    print("Done!")

if __name__ == '__main__':
    main()
