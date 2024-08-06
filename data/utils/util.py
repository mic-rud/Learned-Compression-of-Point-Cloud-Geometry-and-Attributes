import torch
import numpy as np

def custom_collate_fn(data_labels):
    batch ={'points': [], 'colors': [], 'offset': [], 'num_points': [], 'frameIdx': [], 'sequence': [], 'blocks': [], 'voxels': []}
    for item in data_labels:
        if 'points' in item:
            batch['points'].append(item['points'])

        if 'colors' in item:
            batch['colors'].append(item['colors'])

        if 'voxels' in item:
            batch['voxels'].append(item['voxels'])
            batch['voxels'] = torch.stack(batch['voxels'])
        
        batch['offset'].append(item['offset'])
        batch['num_points'].append(item['num_points'])
        batch['frameIdx'].append(item['frameIdx'])
        batch['sequence'].append(item['sequence'])
        batch['blocks'].append(item['blocks'])

    batch['offset'] = torch.stack(batch['offset'])
    batch['num_points'] = torch.stack(batch['num_points'])

    return batch