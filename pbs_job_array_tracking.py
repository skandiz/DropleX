import os
import subprocess
import time

pbs_template = """#!/bin/bash
#PBS -l select=1:ncpus=8:mem=8GB
#PBS -l walltime=23:59:00
#PBS -q SAMPLE_cpuQ
#PBS -o {out_file}
#PBS -e {err_file}

source ~/.bashrc
conda activate DropleX
cd DropleX/

python3 main_tracking.py --video {video} --model {model} --steps {step} --interp {interp} --save --start {start} --end {end} --run
"""

job_directory = "pbs_jobs"
os.makedirs(job_directory, exist_ok=True)


video_selection = "sample_video"  # Replace with your actual video selection
model_name = "skandiz_model_rgb"

nframes = 20000
count = 0
for start in [i*nframes for i in range(0, 20)]:
    end = start + nframes
    out_file = os.path.join(job_directory, f"out_{video_selection}_{start}_{end}.out")
    err_file = os.path.join(job_directory, f"err_{video_selection}_{start}_{end}.err")
    
    pbs_script = pbs_template.format(video = video_selection, model = model_name, step = 2, interp = "linear", start = start, end = end, out_file = out_file, err_file = err_file)
    script_filename = os.path.join(job_directory, f"{count}_{video_selection}_{start}_{end}.pbs")
    with open(script_filename, 'w') as script_file:
        script_file.write(pbs_script)
    
    subprocess.run(['qsub', script_filename])
    os.remove(script_filename)
    time.sleep(0.5)
    count += 1