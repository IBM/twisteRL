
import numpy as np
from twisterl.utils import dynamic_import, json_load_tuples
from twisterl.nn.utils import make_sequential
from twisterl.nn.policy import BasicPolicy
from twisterl.rl.algorithm import timed


def test_dynamic_import():
    sqrt = dynamic_import('math.sqrt')
    assert sqrt(9) == 3


def test_json_load_tuples():
    d = {'__tuple_list__': True, 'list': [[1, 2], [3, 4]]}
    assert json_load_tuples(d) == [(1, 2), (3, 4)]


def test_make_sequential():
    seq = make_sequential(3, (2, 1), final_relu=False)
    layers = list(seq)
    assert len(layers) == 3
    assert layers[-1].__class__.__name__ == 'Linear'


def test_basic_policy_predict():
    policy = BasicPolicy(
        [3],
        2,
        embedding_size=4,
        common_layers=(),
        policy_layers=(2,),
        value_layers=(),
        device="cpu",
    )
    import torch
    with torch.no_grad():
        actions, value = policy.predict(np.array([0.1, 0.2, 0.3], dtype=float))
    assert actions.shape == (2,)
    assert np.isclose(actions.sum(), 1.0)
    assert value.shape == (1,)


def test_timed_decorator():
    @timed
    def add(x, y):
        return x + y

    result, elapsed = add(1, 2)
    assert result == 3
    assert elapsed >= 0
