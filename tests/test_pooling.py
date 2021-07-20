from pkrex.models.utils import dynamic_index_maxpool
from pkrex.models.sampling import batch_index
import torch

EXAMPLE_BATCH_SENTENCE_REP = torch.Tensor(
    [
        [[0.6560, 0.2132, 0.7066, 0.4854, 0.0726],
         [0.7175, 0.8030, 0.9249, 0.2472, 0.0234],
         [0.8356, 0.3106, 0.8910, 0.5076, 0.4114],
         [0.4144, 0.7747, 0.4133, 0.9311, 0.5510],
         [0.7300, 0.3907, 0.7159, 0.8576, 0.0262],
         [0.3761, 0.9466, 0.0187, 0.2557, 0.4266],
         [0.6558, 0.1992, 0.6108, 0.3285, 0.3626],
         [0.9588, 0.0366, 0.0720, 0.3028, 0.3276],
         [0.9403, 0.1356, 0.7833, 0.9921, 0.3905],
         [0.1584, 0.0017, 0.5900, 0.8497, 0.8163]],

        [[0.2106, 0.2689, 0.0481, 0.1458, 0.2331],
         [0.9190, 0.0382, 0.2803, 0.0456, 0.0146],
         [0.7371, 0.5453, 0.2461, 0.3471, 0.1386],
         [0.0501, 0.1450, 0.1375, 0.4566, 0.0964],
         [0.5209, 0.8213, 0.7725, 0.7637, 0.3983],
         [0.2832, 0.3196, 0.7640, 0.5156, 0.1395],
         [0.6406, 0.6548, 0.7528, 0.5210, 0.8913],
         [0.3311, 0.3916, 0.8462, 0.6619, 0.1425],
         [0.7680, 0.9203, 0.6665, 0.1048, 0.4337],
         [0.7242, 0.6274, 0.4291, 0.9485, 0.6114]]
    ]
)

# the above batch has two instances, 10 tokens per sentence, and token embeddings of size 5

EXAMPLE_BATCH_ENTITY_OFFSETS = torch.IntTensor(
    [[[1, 2],
      [3, 10],
      [2, 3]],
     [[7, 8],
      [5, 7],
      [0, 0]]]
)

EXPECTED_POOLING = torch.Tensor(
    [
        [
            [0.7175, 0.8030, 0.9249, 0.2472, 0.0234],  # single token from 1 to 2 (idx1)
            [0.9588, 0.9466, 0.7833, 0.9921, 0.8163],  # maxes element-wise from token 3 to 10
            [0.8356, 0.3106, 0.8910, 0.5076, 0.4114]  # second token
        ],
        [
            [0.3311, 0.3916, 0.8462, 0.6619, 0.1425],
            [0.6406, 0.6548, 0.7640, 0.5210, 0.8913],
            [float('-inf'),  float('-inf'),  float('-inf'),  float('-inf'),  float('-inf')]
        ]
    ]
)


# the above has two instances, 3 maximum entities per sentence and start and end tokens of the entities

def test_dynamic_index_maxpool():
    result = dynamic_index_maxpool(sentence_rep_batch=EXAMPLE_BATCH_SENTENCE_REP,
                                   indices_tensor=EXAMPLE_BATCH_ENTITY_OFFSETS)

    assert torch.all(torch.eq(result, EXPECTED_POOLING))


# TEST INDEXING

EXAMPLE_REL_TUPLES = torch.tensor([
    [
        [0, 1],
        [1, 2],
        [0, 2]
    ],
    [
        [0, 1],
        [0, 0],
        [0, 0]
    ]
], dtype=torch.int64)

EXPECTED_ENT_PAIRS = torch.Tensor(
    [
        [
            [0.7175, 0.8030, 0.9249, 0.2472, 0.0234, 0.9588, 0.9466, 0.7833, 0.9921, 0.8163],
            [0.9588, 0.9466, 0.7833, 0.9921, 0.8163, 0.8356, 0.3106, 0.8910, 0.5076, 0.4114],
            [0.7175, 0.8030, 0.9249, 0.2472, 0.0234, 0.8356, 0.3106, 0.8910, 0.5076, 0.4114]
        ],
        [
            [0.3311, 0.3916, 0.8462, 0.6619, 0.1425, 0.6406, 0.6548, 0.7640, 0.5210, 0.8913],
            [0.3311, 0.3916, 0.8462, 0.6619, 0.1425, 0.3311, 0.3916, 0.8462, 0.6619, 0.1425],
            [0.3311, 0.3916, 0.8462, 0.6619, 0.1425, 0.3311, 0.3916, 0.8462, 0.6619, 0.1425]
        ]
    ]
)


def test_indexing_rels():
    int_result = batch_index(tensor=EXPECTED_POOLING, index=EXAMPLE_REL_TUPLES)
    output = torch.flatten(int_result, start_dim=2, end_dim=3)
    assert torch.all(torch.eq(output, EXPECTED_ENT_PAIRS))
