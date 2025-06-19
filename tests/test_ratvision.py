import unittest
from ratvision.renderer import Renderer # Import your Renderer class

class TestRenderer(unittest.TestCase):
    def test_renderer_initialization(self):
        """
        Test that the Renderer initializes correctly with a config.
        """
        config = {"width": 800, "height": 600}
        renderer = Renderer(config=config)
        self.assertEqual(renderer.config, config)

        renderer_no_config = Renderer()
        self.assertEqual(renderer_no_config.config, {})

    def test_render_method_basic(self):
        """
        Test the basic functionality of the render method.
        """
        renderer = Renderer()
        positions = [(0, 0), (1, 1)]
        head_directions = [(0, 1), (1, 0)]
        result = renderer.render(positions, head_directions)
        expected_output = [
            "Rendered item 0: Pos=(0, 0), HeadDir=(0, 1)",
            "Rendered item 1: Pos=(1, 1), HeadDir=(1, 0)"
        ]
        self.assertEqual(result, expected_output)

    def test_render_method_type_error(self):
        """
        Test that render method raises TypeError for invalid inputs.
        """
        renderer = Renderer()
        with self.assertRaises(TypeError):
            renderer.render("not a list", [])
        with self.assertRaises(TypeError):
            renderer.render([], "not a list")

    def test_render_method_value_error(self):
        """
        Test that render method raises ValueError for mismatched list lengths.
        """
        renderer = Renderer()
        positions = [(0, 0)]
        head_directions = [(0, 1), (1, 0)]
        with self.assertRaises(ValueError):
            renderer.render(positions, head_directions)

if __name__ == '__main__':
    unittest.main()


