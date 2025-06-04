import importlib.util
from pathlib import Path


def load_module():
    module_path = Path(__file__).resolve().parents[1] / 'color selector.py'
    spec = importlib.util.spec_from_file_location('color_selector', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_colors_to_hex():
    module = load_module()
    generator = module.ImagePaletteGenerator()
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    hex_colors = generator.colors_to_hex(colors)
    assert hex_colors == ['#ff0000', '#00ff00', '#0000ff']
