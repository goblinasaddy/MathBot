import numpy as np
from main import draw, model

def test_api_key_configured():
    assert model is not None, "Gemini model should be initialized."

def test_draw_function_basic_usage():
    # Simulate index finger up
    fingers = [0, 1, 0, 0, 0]
    lmList = [[0, 0]] * 9 + [[120, 180]]  # index tip at position 8
    info = (fingers, lmList)
    prev_pos = (100, 160)
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

    current_pos, new_canvas = draw(info, prev_pos, canvas)

    assert isinstance(current_pos, tuple), "Current position should be a tuple."
    assert new_canvas.shape == canvas.shape, "Canvas size should remain the same."
    assert not np.array_equal(canvas, new_canvas), "Canvas should be updated when drawing."

def test_canvas_reset_on_clear_command():
    fingers = [1, 1, 1, 1, 1]  # all fingers up
    lmList = [[0, 0]] * 21
    info = (fingers, lmList)

    prev_pos = (100, 100)
    canvas = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

    current_pos, cleared_canvas = draw(info, prev_pos, canvas)

    assert np.count_nonzero(cleared_canvas) == 0, "Canvas should be completely reset."
    assert current_pos is None, "Current position should be None when canvas is cleared."
