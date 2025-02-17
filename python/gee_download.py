import sys,os,glob
import subprocess
from retry import retry
import ee
import logging
import multiprocessing
import requests
import shutil
import pandas as pd
import numpy as np
import datetime
import time
from multiprocessing import freeze_support
# give the table with the coordinates to download in this process
# give the table with the coordinates to download in this process
tablename=sys.argv[1]
year=sys.argv[2]
tmpdir=os.environ.get('TMPDIR')

#df=pd.read_csv(tablename)
input_directory='/'.join(tablename.split('/')[0:-1])
print(f"input_directory: {input_directory}")

#log_dir=input_directory+'/log'
log_dir=f"{input_directory}/log"
os.makedirs(log_dir, exist_ok=True)
#
basename=tablename.replace('.csv','').split('/')[-1]

#
image_outdir=tmpdir+"/images/"
os.makedirs(image_outdir, exist_ok=True)
resample_image_outdir=tmpdir+"/resampled_images/"
os.makedirs(resample_image_outdir, exist_ok=True)
hdf_outdir=tmpdir+"/hdf5"
os.makedirs(hdf_outdir, exist_ok=True)
hdf_name=hdf_outdir+'/'+basename+'_'+str(year)+'.h5'
#
failed_csv_dir=f"{input_directory}/failed_csv"
os.makedirs(failed_csv_dir, exist_ok=True)
failed_log_filename=f"{failed_csv_dir}/{basename}_{year}_failed.csv"
#failed_df=pd.DataFrame({'FID':[],'lon':[],'lat','reason':[]})
# Ensure the log directory exists
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    print(f"Created log directory: {log_dir}", flush=True)

# Define the log file paths
log_message_txt=f"{log_dir}/output_{basename}_{year}.txt"
stdout_log_path = f"{log_dir}/output_{basename}_{year}.log"
stderr_log_path = f"{log_dir}/error_{basename}_{year}.log"

print(f"stdout log path: {stdout_log_path}", flush=True)
print(f"stderr log path: {stderr_log_path}", flush=True)

try:
    print("Trying to download images")
    print(f"basename: {basename}")
    print(f"input directory: {input_directory}")
    print(f"Log directory: {log_dir}")
    print(f"Table name: {tablename}")
    print(f"Year: {year}")
    print(f"Image output directory: {image_outdir}")
    print("trying to download images")
    # Define the log file paths
    print("Running download_chips_multi_single_sr_subprocess.py script")
    start_time = time.time()
    subprocess.run([
        "python", "/cluster/project/ele/shuo/s2_50000/download_chips_multi_single_sr_subprocess.py",
        "--csv_file", tablename,
        "--year", year,
        "--log_meesage_txt",log_message_txt,
        "--output_dir", image_outdir
    ], stdout=open(f"{log_dir}/output_{basename}_{year}.log", "w"),
       stderr=open(f"{log_dir}/error_{basename}_{year}.log", "w"),
       check=True)
    end_time = time.time()
    print(f"First subprocess took {end_time - start_time:.2f} seconds", flush=True)
    # find which x and y failed and write to failed_df with reason
    start_time = time.time()
    subprocess.run([
        "python", "/cluster/project/ele/shuo/s2_50000/collect_downloading_error_subprocess.py",
        "--combined_log_file", stdout_log_path,
        "--output_error_csv_file", failed_log_filename
    ], check=True)
    # download failed
    end_time = time.time()
    print(f"Second subprocess took {end_time - start_time:.2f} seconds", flush=True)
    start_time = time.time()
    print("load images and create hdf5")
    subprocess.run([
        "python", "/cluster/project/ele/shuo/s2_50000/write_h5_4_pred_bash_seq_12bands_subprocess.py",
        "--dir_path", image_outdir,
        "--out_path", resample_image_outdir,
        "--out_h5_path", hdf_name
    ], check=True)
    print("move images to NFS")
    end_time = time.time()
    print(f"Third subprocess took {end_time - start_time:.2f} seconds", flush=True)
    cmd1="rsync -hvrPt --no-perms --no-owner --no-group"
    print(cmd1)
    cmd2=" "+hdf_name+" "
    print(cmd2)
    cmd3="/nfs/ites-eledata.ethz.ch/mnt/eledata/euler/af_600/"
    cmd3="/nfs/ites-eledata.ethz.ch/mnt/eledata/euler/af_600/width10_"+year
    out_nfs= cmd3+'/'
    if not os.path.exists(cmd3):
        os.makedirs(cmd3)
        print(f"Created log directory: {log_dir}", flush=True)
    id=subprocess.run(cmd1+cmd2+cmd3, shell=True)
    print("SUCCESSFULLY PROCESSED",id)
except:
    print("downloading images failed")
