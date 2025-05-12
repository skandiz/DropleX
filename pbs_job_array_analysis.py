import os
import subprocess
import time

pbs_template = """#!/bin/bash
#PBS -l select=1:ncpus=24:mem=40GB
#PBS -l walltime=5:59:00
#PBS -q SAMPLE_cpuQ
#PBS -o {out_file}
#PBS -e {err_file}

source ~/.bashrc
conda activate DropleX
cd DropleX/

python3 main_analysis.py --trajectory={trajectory} --steps={steps} --run
"""

job_directory = "pbs_jobs"
os.makedirs(job_directory, exist_ok=True)

for trajectory in ['sample_video']:
    out_file = os.path.join(job_directory, f"out_{trajectory}.out")
    err_file = os.path.join(job_directory, f"err_{trajectory}.err")
    
    pbs_script = pbs_template.format(trajectory = trajectory, steps = 8, out_file = out_file, err_file = err_file)
    script_filename = os.path.join(job_directory, f"{trajectory}.pbs")
    with open(script_filename, 'w') as script_file:
        script_file.write(pbs_script)
    
    subprocess.run(['qsub', script_filename])
    os.remove(script_filename)
    time.sleep(0.5)