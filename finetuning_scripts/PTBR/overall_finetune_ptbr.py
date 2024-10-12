import subprocess
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"  # Change this depending on the GPU to use

def run_script(script_name):
    try:
        # Run the script
        result = subprocess.run(['python', script_name], check=True, text=True, capture_output=True)
        
        # Define the output file path
        output_dir = os.path.expanduser("~/tese/results/PTBR")
        output_file_path = os.path.join(output_dir, f"{script_name}_output.txt")
        
        # Create the directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Write the output to the file
        with open(output_file_path, 'w') as output_file:
            output_file.write(result.stdout)
        
        # Print the output to the console
        print(f"Output of {script_name}:")
        print(result.stdout)

    except subprocess.CalledProcessError as e:
        # Define the error file path
        error_dir = os.path.expanduser("~/tese/results/PTBR")
        error_file_path = os.path.join(error_dir, f"{script_name}_error.txt")
        
        # Create the directory if it doesn't exist
        os.makedirs(error_dir, exist_ok=True)
        
        # Write the error to the file
        with open(error_file_path, 'w') as error_file:
            error_file.write(e.stderr)
        
        # Print the error to the console
        print(f"Error occurred while running {script_name}:")
        print(e.stderr)

if __name__ == "__main__":
    scripts = [
        'finetune_boolq_ptbr.py',    # BooLQ
        'finetune_cb_ptbr.py',       # CB
        'finetune_copa_ptbr.py',     # COPA
        'finetune_mrpc_ptbr.py',     # MRPC
        'finetune_multirc_ptbr.py',  # MultiRC
        'finetune_rte_ptbr.py',      # RTE
        'finetune_sst2_ptbr.py',     # SST2
        'finetune_stsb_ptbr.py',     # STS-B
        'finetune_wic_ptbr.py',      # WiC
        'finetune_wnli_ptbr.py',     # WNLI
        'finetune_wsc_ptbr.py',      # WSC
    ]
    
    for script in scripts:
        run_script(script)
