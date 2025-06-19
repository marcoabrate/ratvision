import json
from renderer import Renderer

with open('/Users/marco/Downloads/positions.json', 'r') as f:
    positions = json.load(f)
with open('/Users/marco/Downloads/head_directions.json', 'r') as f:
    head_directions = json.load(f)

r = Renderer(
    config={
        'blender_dir': '/Applications/Blender.app/Contents/MacOS/Blender',
        'env_file': '/Users/marco/phd/ratvision/ratvision/environments/box_messy.blend'
    }
)
r.render(positions, head_directions)
