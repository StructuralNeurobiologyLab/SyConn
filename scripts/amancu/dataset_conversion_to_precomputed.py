import os, sys
import numpy as np
import multiprocessing
from knossos_utils import KnossosDataset
from PIL import Image
from cloudvolume import CloudVolume
from cloudvolume.lib import mkdir, touch
from concurrent.futures import ProcessPoolExecutor
import time

kd = KnossosDataset("/wholebrain/songbird/j0251/j0251_72_clahe2/")

volume_info = {
        "type": "neuroglancer_multiscale_volume",
        "layer_type": "image",
        "data_type": "uint8",
        "num_channels": 1,
        "scales": [
            {
                "key": "1_1_1",
                "size": [512,1024,1024],#[27136, 27392, 15616],
                "resolution": [10, 10, 25],
                "chunk_sizes": [[64, 64, 64]],
                "encoding": "jpeg",
            },
            {
                "key": "2_2_2",
                "size": [256,512,512],#[13568, 13696, 7808],
                "resolution": [20, 20, 50],
                "chunk_sizes": [[64, 64, 64]],
                "encoding": "jpeg",
            },
            # {
            #     "key": "4_4_4",
            #     "size": [6784, 6912, 3968],
            #     "resolution": [40, 40, 100],
            #     "chunk_sizes": [[64, 64, 64]],
            #     "encoding": "jpeg",
            # },
            # {
            #     "key": "8_8_8",
            #     "size": [3389, 3418, 1936],
            #     "resolution": [80, 80, 200],
            #     "chunk_sizes": [[64, 64, 64]],
            #     "encoding": "jpeg",
            # },
            # {
            #     "key": "16_16_16",
            #     "size": [1694, 1709, 968],
            #     "resolution": [160, 160, 400],
            #     "chunk_sizes": [[64, 64, 64]],
            #     "encoding": "jpeg",
            # },
            # {
            #     "key": "32_32_32",
            #     "size": [847, 854, 484],
            #     "resolution": [320, 320, 800],
            #     "chunk_sizes": [[64, 64, 64]],
            #     "encoding": "jpeg",
            # },
            # {
            #     "key": "64_64_64",
            #     "size": [423, 427, 242],
            #     "resolution": [640, 640, 1600],
            #     "chunk_sizes": [[64, 64, 64]],
            #     "encoding": "jpeg",
            # }
        ]
    }

try:
  # vol = CloudVolume('file:///wholebrain/songbird/j0251/j0251_72_clahe2_precomputed/', info=volume_info, bounded=False, non_aligned_writes=True)
  vol = CloudVolume('file:///wholebrain/scratch/amancu/j0251/example', info=volume_info, bounded=False, non_aligned_writes=True)
  vol.provenance.description = 'Whole j0251 dataset conversion'
  vol.provenance.owners = ['kornfeld@neuro.mpg.de','amancu@neuro.mpg.de', 'hashir@neuro.mpg.de']  # list of contact email addresses

  vol.commit_info()  # generates file://bucket/dataset/layer/info json file
  vol.commit_provenance()  # generates file://bucket/dataset/layer/provenance json file

  to_upload = []

  # split dataset
  num_x, num_y, num_z = kd.boundary[0]//512, kd.boundary[1]//512, kd.boundary[2]//512
  print(f'Splitting dataset into number of chunks on each xyz-axis: {num_x}, {num_y}, {num_z}')

  for x in range(num_x):
    for y in range(num_y):
      for z in range(num_z):
        to_upload.append((x*512,y*512,z*512))

  to_upload = [(2048,2048,2048), (2048,2560,2048), (2048,2048,2560), (2048,2560,2560)]

except IOError as err:
  errno, strerror = err.args
  print ('I/O error({0}): {1}'.format(errno, strerror))
  print (err)
except ValueError as ve:
  print ('Could not convert data to an integer.')
  print (ve) 
except:
  print ('Unexpected error:', sys.exc_info()[0])
  raise

def process(z):
    global kd
    try:
      x_start = z[0] 
      y_start = z[1]
      z_start = z[2]
      x_end = x_start + 512 if (x_start + 512) < kd.boundary[0] else kd.boundary[0]
      y_end = y_start + 512 if (y_start + 512) < kd.boundary[1] else kd.boundary[1]
      z_end = z_start + 512 if (z_start + 512) < kd.boundary[2] else kd.boundary[2]

      print(f"start {x_start} {y_start} {z_start}")
      print(f"end {x_end} {y_end} {z_end}")
      
      size = (x_end-x_start, y_end-y_start, z_end-z_start)

      # get bytes array from Knossos and reshape it to the according required size
      bytes_array = kd.load_raw(offset=(x_start, y_start, z_start), size=size, mag=1)
      array = bytes_array.reshape(size)
      array = np.swapaxes(array,0,2)

      vol[x_start-2048:x_end-2048, y_start-2048:y_end-2048, z_start-2048:z_end-2048] = array
      # vol[x_start:x_end, y_start:y_end, z_start:z_end] = array
      print('put in array')
    except IOError as err:
      errno, strerror = err.args
      print ('I/O error({0}): {1}'.format(errno, strerror))
      print (err)
    except ValueError as ve:
      print ('Could not convert data to an integer.')
      print (ve) 
    except:
      print ('Unexpected error:', sys.exc_info()[0])
      raise

with ProcessPoolExecutor(max_workers=2) as executor:
    executor.map(process, to_upload)
  
time.sleep(2)
img = vol[0:512,0:1020,0:1021]
img.viewer()