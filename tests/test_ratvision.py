import unittest
from ratvision.renderer import Renderer

class TestRenderer(unittest.TestCase):
    def test_renderer_initialization(self):
        '''
        Test that the Renderer initializes correctly with updated config.
        '''
        config = {'output_dir': 'new/output/dir', 'camera_height': 0.04}
        renderer = Renderer(blender_exec='', config=config)

        keys = config.keys()
        self.assertEqual([renderer.config[k] for k in keys], [config[k] for k in keys])

        renderer_no_config = Renderer(blender_exec='')
        self.assertEqual(
            [c for c in renderer_no_config.config if 'output' not in c],
            [c for c in Renderer.DEFAULT_CONFIG if 'output' not in c]
        )

    def test_render_method_type_error(self):
        '''
        Test that render method raises TypeError for invalid inputs.
        '''
        renderer = Renderer(blender_exec='')
        with self.assertRaises(TypeError):
            renderer.render('not a list', [])
        with self.assertRaises(TypeError):
            renderer.render([], 'not a list')

    def test_render_method_value_error(self):
        '''
        Test that render method raises ValueError for mismatched list lengths.
        '''
        renderer = Renderer(blender_exec='')
        positions = [(0, 0)]
        head_directions = [0, 0]
        with self.assertRaises(ValueError):
            renderer.render(positions, head_directions)

if __name__ == '__main__':
    unittest.main()


