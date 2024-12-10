pip install --upgrade open3d
python extract_mesh_tsdf.py -m outputs/dtu_1001/scan106 --iteration 30000
pip install open3d==0.10.0
python evaluate_dtu_mesh.py -m outputs/dtu_1001/scan106 --iteration 30000 --DTU /mnt/workfiles/datasets/dtu/SampleSet/MVS_Data
