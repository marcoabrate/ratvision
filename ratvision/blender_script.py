import bpy
import json
import os
import platform
import sys

argv = sys.argv
argv = argv[argv.index("--") + 1:]

# get the paths to the temporary files generated by main script
# and passed as command line arguments
positions_file = os.path.normpath(argv[0].strip())
head_directions_file = os.path.normpath(argv[1].strip())
config_file = os.path.normpath(argv[2].strip())

# load the positions, head_directions, and config from temporary files
with open(positions_file, 'r') as f:
    positions = json.load(f)
with open(head_directions_file, 'r') as f:
    head_directions = json.load(f)
with open(config_file, 'r') as f:
    config = json.load(f)

# define the camera object, which should be present in the Blender scene
# and initialize some key parameters
camera =  bpy.data.objects[config['camera_name']]

camera_height = config['camera_height']
camera_vertical_angle = config['camera_vertical_angle']

# initialize frame count and define the camera's trajectory
frame_count = 1
for p, hd in zip(positions, head_directions):

    # update the camera's position
    camera.location = (p[0], p[1], camera_height)
    camera.keyframe_insert(data_path="location", frame=frame_count)
    
    # and orientation
    camera.rotation_euler = (camera_vertical_angle, 0, hd)
    camera.keyframe_insert(data_path="rotation_euler", frame=frame_count)
    
    frame_count += 1
    
############################
### RENDERING PARAMETERS ###
############################

scene = bpy.context.scene

# cycles is needed when rendering in "panoramic" camera mode
# you can change this to "BLENDER_EEVEE" if you changed camera mode
scene.render.engine = "CYCLES"

# set the preferred compute device to GPU (or METAL on Mac)
# this falls back to CPU if GPU is not available
if 'mac' in platform.platform():
    bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "METAL"
else:
    bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA" # "CUDA" or "OPTIX"
    bpy.context.scene.cycles.denoiser = "OPTIX"
bpy.context.scene.cycles.device = "GPU"
bpy.context.preferences.addons["cycles"].preferences.refresh_devices()

# set the frames' x and y resolutions
scene.render.resolution_x = config['frame_dim'][0]
scene.render.resolution_y = config['frame_dim'][1]
