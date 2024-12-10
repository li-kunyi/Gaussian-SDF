python extract_mesh_gsdf.py -m outputs/dtu_1001/scan24 --iteration 30000
python extract_mesh_gsdf.py -m outputs/dtu_1001/scan37 --iteration 30000
python extract_mesh_gsdf.py -m outputs/dtu_1001/scan40 --iteration 30000
python extract_mesh_gsdf.py -m outputs/dtu_1001/scan55 --iteration 30000
python extract_mesh_gsdf.py -m outputs/dtu_1001/scan63 --iteration 30000
python extract_mesh_gsdf.py -m outputs/dtu_1001/scan65 --iteration 30000
python extract_mesh_gsdf.py -m outputs/dtu_1001/scan69 --iteration 30000
python extract_mesh_gsdf.py -m outputs/dtu_1001/scan83 --iteration 30000
python extract_mesh_gsdf.py -m outputs/dtu_1001/scan97 --iteration 30000
python extract_mesh_gsdf.py -m outputs/dtu_1001/scan105 --iteration 30000
python extract_mesh_gsdf.py -m outputs/dtu_1001/scan106 --iteration 30000
python extract_mesh_gsdf.py -m outputs/dtu_1001/scan110 --iteration 30000
python extract_mesh_gsdf.py -m outputs/dtu_1001/scan114 --iteration 30000
python extract_mesh_gsdf.py -m outputs/dtu_1001/scan118 --iteration 30000
python extract_mesh_gsdf.py -m outputs/dtu_1001/scan122 --iteration 30000

pip install --upgrade open3d
python extract_mesh_tsdf.py -m outputs/dtu_1001/scan24 --iteration 30000
python extract_mesh_tsdf.py -m outputs/dtu_1001/scan37 --iteration 30000
python extract_mesh_tsdf.py -m outputs/dtu_1001/scan40 --iteration 30000
python extract_mesh_tsdf.py -m outputs/dtu_1001/scan55 --iteration 30000
python extract_mesh_tsdf.py -m outputs/dtu_1001/scan63 --iteration 30000
python extract_mesh_tsdf.py -m outputs/dtu_1001/scan65 --iteration 30000
python extract_mesh_tsdf.py -m outputs/dtu_1001/scan69 --iteration 30000
python extract_mesh_tsdf.py -m outputs/dtu_1001/scan83 --iteration 30000
python extract_mesh_tsdf.py -m outputs/dtu_1001/scan97 --iteration 30000
python extract_mesh_tsdf.py -m outputs/dtu_1001/scan105 --iteration 30000
python extract_mesh_tsdf.py -m outputs/dtu_1001/scan106 --iteration 30000
python extract_mesh_tsdf.py -m outputs/dtu_1001/scan110 --iteration 30000
python extract_mesh_tsdf.py -m outputs/dtu_1001/scan114 --iteration 30000
python extract_mesh_tsdf.py -m outputs/dtu_1001/scan118 --iteration 30000
python extract_mesh_tsdf.py -m outputs/dtu_1001/scan122 --iteration 30000
pip install open3d==0.10.0

python evaluate_dtu_mesh.py -m outputs/dtu_1001/scan24 --iteration 30000 --DTU /mnt/workfiles/datasets/dtu/SampleSet/MVS_Data
python evaluate_dtu_mesh.py -m outputs/dtu_1001/scan37 --iteration 30000 --DTU /mnt/workfiles/datasets/dtu/SampleSet/MVS_Data
python evaluate_dtu_mesh.py -m outputs/dtu_1001/scan40 --iteration 30000 --DTU /mnt/workfiles/datasets/dtu/SampleSet/MVS_Data
python evaluate_dtu_mesh.py -m outputs/dtu_1001/scan55 --iteration 30000 --DTU /mnt/workfiles/datasets/dtu/SampleSet/MVS_Data
python evaluate_dtu_mesh.py -m outputs/dtu_1001/scan63 --iteration 30000 --DTU /mnt/workfiles/datasets/dtu/SampleSet/MVS_Data
python evaluate_dtu_mesh.py -m outputs/dtu_1001/scan65 --iteration 30000 --DTU /mnt/workfiles/datasets/dtu/SampleSet/MVS_Data
python evaluate_dtu_mesh.py -m outputs/dtu_1001/scan69 --iteration 30000 --DTU /mnt/workfiles/datasets/dtu/SampleSet/MVS_Data
python evaluate_dtu_mesh.py -m outputs/dtu_1001/scan83 --iteration 30000 --DTU /mnt/workfiles/datasets/dtu/SampleSet/MVS_Data
python evaluate_dtu_mesh.py -m outputs/dtu_1001/scan97 --iteration 30000 --DTU /mnt/workfiles/datasets/dtu/SampleSet/MVS_Data
python evaluate_dtu_mesh.py -m outputs/dtu_1001/scan105 --iteration 30000 --DTU /mnt/workfiles/datasets/dtu/SampleSet/MVS_Data
python evaluate_dtu_mesh.py -m outputs/dtu_1001/scan106 --iteration 30000 --DTU /mnt/workfiles/datasets/dtu/SampleSet/MVS_Data
python evaluate_dtu_mesh.py -m outputs/dtu_1001/scan110 --iteration 30000 --DTU /mnt/workfiles/datasets/dtu/SampleSet/MVS_Data
python evaluate_dtu_mesh.py -m outputs/dtu_1001/scan114 --iteration 30000 --DTU /mnt/workfiles/datasets/dtu/SampleSet/MVS_Data
python evaluate_dtu_mesh.py -m outputs/dtu_1001/scan118 --iteration 30000 --DTU /mnt/workfiles/datasets/dtu/SampleSet/MVS_Data
python evaluate_dtu_mesh.py -m outputs/dtu_1001/scan122 --iteration 30000 --DTU /mnt/workfiles/datasets/dtu/SampleSet/MVS_Data