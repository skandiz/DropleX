import os
import subprocess
import time

pbs_template = """#!/bin/bash
#PBS -l select=1:ncpus=16:mem=20GB
#PBS -l walltime=5:59:00
#PBS -q VARIAMOLS_cpuQ
#PBS -o {out_file}
#PBS -e {err_file}

source ~/.bashrc
conda activate DropleX
cd projects/abp_project_cluster/

python3 main_analysis.py --trajectory={trajectory} --steps={steps}
"""
#PBS -M matteo.scandola@unitn.it
#PBS -m abe

job_directory = "pbs_jobs"
os.makedirs(job_directory, exist_ok=True)
for trajectory in ['25b25r_lowconc_2', '25b25r_lowconc_5', '25b25r_lowconc_6', 'density_test_3_1', 'density_test_3_2', 'density_test_4_1', 'density_test_4_2', 'density_test_5_1', 'density_test_5_2', 'density_test_8_1', 'density_test_8_2', 'density_test_9_2',  'density_test_10_1', 'density_test_12_1', 'density_test_13_1', 'density_test_17']:
    out_file = os.path.join(job_directory, f"out_{trajectory}.out")
    err_file = os.path.join(job_directory, f"err_{trajectory}.err")
    
    pbs_script = pbs_template.format(trajectory = trajectory, steps = 9, out_file = out_file, err_file = err_file)
    script_filename = os.path.join(job_directory, f"{trajectory}.pbs")
    with open(script_filename, 'w') as script_file:
        script_file.write(pbs_script)
    
    subprocess.run(['qsub', script_filename])
    os.remove(script_filename)
    time.sleep(0.5)