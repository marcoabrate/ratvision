import bpy
import json
import sys
import math
import platform

argv = sys.argv
argv = argv[argv.index("--") + 1:]

positions_file = argv[0].strip()
head_directions_file = argv[1].strip()
config_file = argv[2].strip()

with open(positions_file, 'r') as f:
    positions = json.load(f)
with open(head_directions_file, 'r') as f:
    head_directions = json.load(f)
with open(config_file, 'r') as f:
    config = json.load(f)

camera =  bpy.data.objects[config['camera_name']]

frame_count = 1
camera_height = config['camera_height']
camera_vertical_angle = config['camera_vertical_angle']*math.pi/180

for p, hd in zip(positions, head_directions):        
    x, y = p[0], p[1]

    camera.location = (x, y, camera_height)
    camera.keyframe_insert(data_path="location", frame=frame_count)
    
    camera.rotation_euler = (camera_vertical_angle, 0, hd*math.pi/180)
    camera.keyframe_insert(data_path="rotation_euler", frame=frame_count)
    
    frame_count += 1
    
########################
### RENDERING PARAMETERS
########################

scene = bpy.context.scene

scene.render.engine = "CYCLES"

# Set the device_type
if 'mac' in platform.platform():
    bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "METAL"
else:
    bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA" # "CUDA" or "OPTIX"
    bpy.context.scene.cycles.denoiser = "OPTIX"

bpy.context.scene.cycles.device = "GPU"
bpy.context.preferences.addons["cycles"].preferences.refresh_devices()

# Resolution change
scene.render.resolution_x = config['frame_dim'][0]
scene.render.resolution_y = config['frame_dim'][1]
