import sys
sys.path.insert(0, 'src')

from pose_uncertainty.data_loader import COCOLoader

print('='*60)
print('VERIFICANDO DATA LOADER COCO')
print('='*60)

loader = COCOLoader(
    data_root='data/coco',
    ann_file='annotations/person_keypoints_val2017.json',
    image_dir='val2017',
    min_keypoints=5
)

print(f'\nTotal imagenes: {len(loader)}')

for i, sample in enumerate(loader):
    if i >= 2:
        break
    print(f'\n--- Muestra {i+1} ---')
    print(f'ID: {sample.image_id}')
    print(f'Shape: {sample.image_array.shape}')
    visible = sum(1 for kp in sample.ground_truth_keypoints if kp.confidence > 0.5)
    print(f'Keypoints visibles: {visible}/17')
    print('Primeros 3 keypoints:')
    for kp in sample.ground_truth_keypoints[:3]:
        vis = 'visible' if kp.confidence == 1.0 else 'occluded' if kp.confidence == 0.5 else 'not labeled'
        print(f'  {kp.name}: ({kp.x:.1f}, {kp.y:.1f}) [{vis}]')

print('\n' + '='*60)
print('PRUEBA COMPLETADA')
print('='*60)
