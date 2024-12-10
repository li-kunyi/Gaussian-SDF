ulimit -n 4096

# TNT
# python train_sdf_depth.py -s /mnt/workfiles/datasets/tnt/TrainingSet/Barn -m outputs/tnt_1011/Barn -r 2 --eval --save_ckpt --use_decoupled_appearance #--ckpt_pth outputs/tnt_1011/Barn/ckpt/ckpt_20000.pth
# python extract_mesh_gsdf.py -m outputs/tnt_1011/Barn --iteration 30000
# # pip install --upgrade open3d
# # python extract_mesh_tsdf.py -m outputs/tnt_1011/Barn --iteration 30000
# # pip install open3d==0.10.0
# python eval_tnt/run.py --dataset-dir eval_tnt/TrainingSet/Barn --traj-path /mnt/workfiles/datasets/tnt/TrainingSet/Barn/Barn_COLMAP_SfM.log --ply-path outputs/tnt_1011/Barn/test/ours_30000/fusion/mesh_binary_search_7.ply

# python train_sdf_depth.py -s /mnt/workfiles/datasets/tnt/TrainingSet/Caterpillar -m outputs/tnt_1011/Caterpillar -r 2 --eval --save_ckpt --use_decoupled_appearance
# python extract_mesh_gsdf.py -m outputs/tnt_1011/Caterpillar --iteration 30000
# # pip install --upgrade open3d
# # python extract_mesh_tsdf.py -m outputs/tnt_1011/Caterpillar --iteration 30000
# # pip install open3d==0.10.0
# python eval_tnt/run.py --dataset-dir eval_tnt/TrainingSet/Caterpillar --traj-path /mnt/workfiles/datasets/tnt/TrainingSet/Caterpillar/Caterpillar_COLMAP_SfM.log --ply-path outputs/tnt_1011/Caterpillar/test/ours_30000/fusion/mesh_binary_search_7.ply

# python train_sdf_depth.py -s /mnt/workfiles/datasets/tnt/TrainingSet/Courthouse -m outputs/tnt_1011/Courthouse -r 2 --eval --save_ckpt --use_decoupled_appearance
# python extract_mesh_gsdf.py -m outputs/tnt_1011/Courthouse --iteration 30000
# # pip install --upgrade open3d
# # python extract_mesh_tsdf.py -m outputs/tnt_1011/Courthouse --iteration 30000
# # pip install open3d==0.10.0
# python eval_tnt/run.py --dataset-dir eval_tnt/TrainingSet/Courthouse --traj-path /mnt/workfiles/datasets/tnt/TrainingSet/Courthouse/Courthouse_COLMAP_SfM.log --ply-path outputs/tnt_1011/Courthouse/test/ours_30000/fusion/mesh_binary_search_7.ply

# python train_sdf_depth.py -s /mnt/workfiles/datasets/tnt/TrainingSet/Ignatius -m outputs/tnt_1011/Ignatius -r 2 --eval --save_ckpt --use_decoupled_appearance
# python extract_mesh_gsdf.py -m outputs/tnt_1011/Ignatius --iteration 30000
# # pip install --upgrade open3d
# # python extract_mesh_tsdf.py -m outputs/tnt_1011/Ignatius --iteration 30000
# # pip install open3d==0.10.0
# python eval_tnt/run.py --dataset-dir eval_tnt/TrainingSet/Ignatius --traj-path /mnt/workfiles/datasets/tnt/TrainingSet/Ignatius/Ignatius_COLMAP_SfM.log --ply-path outputs/tnt_1011/Ignatius/test/ours_30000/fusion/mesh_binary_search_7.ply

# python train_sdf_depth.py -s /mnt/workfiles/datasets/tnt/TrainingSet/Meetingroom -m outputs/tnt_1011/Meetingroom -r 2 --eval --save_ckpt --use_decoupled_appearance
# python extract_mesh_gsdf.py -m outputs/tnt_1011/Meetingroom --iteration 30000
# # pip install --upgrade open3d
# # python extract_mesh_tsdf.py -m outputs/tnt_1011/Meetingroom --iteration 30000
# # pip install open3d==0.10.0
# python eval_tnt/run.py --dataset-dir eval_tnt/TrainingSet/Meetingroom --traj-path /mnt/workfiles/datasets/tnt/TrainingSet/Meetingroom/Meetingroom_COLMAP_SfM.log --ply-path outputs/tnt_1011/Meetingroom/test/ours_30000/fusion/mesh_binary_search_7.ply

# python train_sdf_depth.py -s /mnt/workfiles/datasets/tnt/TrainingSet/Truck -m outputs/tnt_1011/Truck -r 2 --eval --save_ckpt --use_decoupled_appearance
# python extract_mesh_gsdf.py -m outputs/tnt_1011/Truck --iteration 30000
# # # pip install --upgrade open3d
# # # python extract_mesh_tsdf.py -m outputs/tnt_1011/Truck --iteration 30000
# # # pip install open3d==0.10.0
# python eval_tnt/run.py --dataset-dir eval_tnt/TrainingSet/Truck --traj-path /mnt/workfiles/datasets/tnt/TrainingSet/Truck/Truck_COLMAP_SfM.log --ply-path outputs/tnt_1011/Truck/test/ours_30000/fusion/mesh_binary_search_7.ply

# MIP 360
# python train_sdf_depth.py -s /mnt/workfiles/datasets/360/garden -m outputs/360_1011/garden -r 4 --eval --save_ckpt --lambda_depth 0.01 --lambda_normal 0.01
# python render.py -m outputs/360_1011/garden --data_device cpu --skip_train
# python metrics.py -m outputs/360_1011/garden -r 4

# python train_sdf_depth.py -s /mnt/workfiles/datasets/360/bicycle -m outputs/360_1011/bicycle -r 4 --eval --save_ckpt --lambda_depth 0.01 --lambda_normal 0.01
# python render.py -m outputs/360_1011/bicycle --data_device cpu --skip_train
# python metrics.py -m outputs/360_1011/bicycle -r 4

# python train_sdf_depth.py -s /mnt/workfiles/datasets/360/stump -m outputs/360_1011/stump -r 4 --eval --save_ckpt --lambda_depth 0.01 --lambda_normal 0.01
# python render.py -m outputs/360_1011/stump --data_device cpu --skip_train
# python metrics.py -m outputs/360_1011/stump -r 4

# python train_sdf_depth.py -s /mnt/workfiles/datasets/360/room -m outputs/360_1011/room -r 2 --eval --save_ckpt --lambda_depth 0.01 --lambda_normal 0.01
# python render.py -m outputs/360_1011/room --data_device cpu --skip_train
# python metrics.py -m outputs/360_1011/room -r 2

# python train_sdf_depth.py -s /mnt/workfiles/datasets/360/bonsai -m outputs/360_1011/bonsai -r 2 --eval --save_ckpt --lambda_depth 0.01 --lambda_normal 0.01
# python render.py -m outputs/360_1011/bonsai --data_device cpu --skip_train
# python metrics.py -m outputs/360_1011/bonsai -r 2

# python train_sdf_depth.py -s /mnt/workfiles/datasets/360/counter -m outputs/360_1011/counter -r 2 --eval --save_ckpt --lambda_depth 0.01 --lambda_normal 0.01
# python render.py -m outputs/360_1011/counter --data_device cpu --skip_train
# python metrics.py -m outputs/360_1011/counter -r 2

# python train_sdf_depth.py -s /mnt/workfiles/datasets/360/kitchen -m outputs/360_1011/kitchen -r 2 --eval --save_ckpt --lambda_depth 0.01 --lambda_normal 0.01
# python render.py -m outputs/360_1011/kitchen --data_device cpu --skip_train
# python metrics.py -m outputs/360_1011/kitchen -r 2


# dtu
# python train_sdf_depth.py -s /mnt/workfiles/datasets/dtu/scan24 -m outputs/dtu_new/scan24 -r 2 --eval --save_ckpt --use_decoupled_appearance --lambda_distortion 1000 --lambda_depth 0.01 --lambda_normal 0.01
# pip install --upgrade open3d
# python extract_mesh_tsdf.py -m outputs/dtu_new/scan24 --iteration 30000
# pip install open3d==0.10.0
# python evaluate_dtu_mesh.py -m outputs/dtu_new/scan24 --iteration 30000 --DTU /mnt/workfiles/datasets/dtu/SampleSet/MVS_Data

# python train_sdf_depth.py -s /mnt/workfiles/datasets/dtu/scan37 -m outputs/dtu_new/scan37 -r 2 --eval --save_ckpt --use_decoupled_appearance --lambda_distortion 1000 --lambda_depth 0.01 --lambda_normal 0.01
# pip install --upgrade open3d
# python extract_mesh_tsdf.py -m outputs/dtu_new/scan37 --iteration 30000
# pip install open3d==0.10.0
# python evaluate_dtu_mesh.py -m outputs/dtu_new/scan37 --iteration 30000 --DTU /mnt/workfiles/datasets/dtu/SampleSet/MVS_Data

# python train_sdf_depth.py -s /mnt/workfiles/datasets/dtu/scan40 -m outputs/dtu_new/scan40 -r 2 --eval --save_ckpt --use_decoupled_appearance --lambda_distortion 1000 --lambda_depth 0.01 --lambda_normal 0.01
# pip install --upgrade open3d
# python extract_mesh_tsdf.py -m outputs/dtu_new/scan40 --iteration 30000
# pip install open3d==0.10.0
# python evaluate_dtu_mesh.py -m outputs/dtu_new/scan40 --iteration 30000 --DTU /mnt/workfiles/datasets/dtu/SampleSet/MVS_Data

# python train_sdf_depth.py -s /mnt/workfiles/datasets/dtu/scan55 -m outputs/dtu_new/scan55 -r 2 --eval --save_ckpt --use_decoupled_appearance --lambda_distortion 1000 --lambda_depth 0.01 --lambda_normal 0.01
# pip install --upgrade open3d
# python extract_mesh_tsdf.py -m outputs/dtu_new/scan55 --iteration 30000
# pip install open3d==0.10.0
# python evaluate_dtu_mesh.py -m outputs/dtu_new/scan55 --iteration 30000 --DTU /mnt/workfiles/datasets/dtu/SampleSet/MVS_Data

# python train_sdf_depth.py -s /mnt/workfiles/datasets/dtu/scan63 -m outputs/dtu_new/scan63 -r 2 --eval --save_ckpt --use_decoupled_appearance --lambda_distortion 1000 --lambda_depth 0.01 --lambda_normal 0.01 --ckpt_pth outputs/dtu_new/scan63/ckpt/ckpt_20000.pth
# pip install --upgrade open3d
# python extract_mesh_tsdf.py -m outputs/dtu_new/scan63 --iteration 30000
# pip install open3d==0.10.0
# python evaluate_dtu_mesh.py -m outputs/dtu_new/scan63 --iteration 30000 --DTU /mnt/workfiles/datasets/dtu/SampleSet/MVS_Data

# python train_sdf_depth.py -s /mnt/workfiles/datasets/dtu/scan65 -m outputs/dtu_new/scan65 -r 2 --eval --save_ckpt --use_decoupled_appearance --lambda_distortion 1000 --lambda_depth 0.01 --lambda_normal 0.01
# pip install --upgrade open3d
# python extract_mesh_tsdf.py -m outputs/dtu_new/scan65 --iteration 30000
# pip install open3d==0.10.0
# python evaluate_dtu_mesh.py -m outputs/dtu_new/scan65 --iteration 30000 --DTU /mnt/workfiles/datasets/dtu/SampleSet/MVS_Data

# python train_sdf_depth.py -s /mnt/workfiles/datasets/dtu/scan69 -m outputs/dtu_new/scan69 -r 2 --eval --save_ckpt --use_decoupled_appearance --lambda_distortion 1000 --lambda_depth 0.01 --lambda_normal 0.01
# pip install --upgrade open3d
# python extract_mesh_tsdf.py -m outputs/dtu_new/scan69 --iteration 30000
# pip install open3d==0.10.0
# python evaluate_dtu_mesh.py -m outputs/dtu_new/scan69 --iteration 30000 --DTU /mnt/workfiles/datasets/dtu/SampleSet/MVS_Data

# python train_sdf_depth.py -s /mnt/workfiles/datasets/dtu/scan83 -m outputs/dtu_new/scan83 -r 2 --eval --save_ckpt --use_decoupled_appearance --lambda_distortion 1000 --lambda_depth 0.01 --lambda_normal 0.01
# pip install --upgrade open3d
# python extract_mesh_tsdf.py -m outputs/dtu_new/scan83 --iteration 30000
# pip install open3d==0.10.0
# python evaluate_dtu_mesh.py -m outputs/dtu_new/scan83 --iteration 30000 --DTU /mnt/workfiles/datasets/dtu/SampleSet/MVS_Data

# python train_sdf_depth.py -s /mnt/workfiles/datasets/dtu/scan97 -m outputs/dtu_new/scan97 -r 2 --eval --save_ckpt --use_decoupled_appearance --lambda_distortion 1000 --lambda_depth 0.01 --lambda_normal 0.01
# pip install --upgrade open3d
# python extract_mesh_tsdf.py -m outputs/dtu_new/scan97 --iteration 30000
# pip install open3d==0.10.0
# python evaluate_dtu_mesh.py -m outputs/dtu_new/scan97 --iteration 30000 --DTU /mnt/workfiles/datasets/dtu/SampleSet/MVS_Data

# python train_sdf_depth.py -s /mnt/workfiles/datasets/dtu/scan105 -m outputs/dtu_new/scan105 -r 2 --eval --save_ckpt --use_decoupled_appearance --lambda_distortion 1000 --lambda_depth 0.01 --lambda_normal 0.01
# pip install --upgrade open3d
# python extract_mesh_tsdf.py -m outputs/dtu_new/scan105 --iteration 30000
# pip install open3d==0.10.0
# python evaluate_dtu_mesh.py -m outputs/dtu_new/scan105 --iteration 30000 --DTU /mnt/workfiles/datasets/dtu/SampleSet/MVS_Data

python train_sdf_depth.py -s /mnt/workfiles/datasets/dtu/scan106 -m outputs/dtu_new/scan106 -r 2 --eval --save_ckpt --use_decoupled_appearance --lambda_distortion 1000 --lambda_depth 0.01 --lambda_normal 0.01
pip install --upgrade open3d
python extract_mesh_tsdf.py -m outputs/dtu_new/scan106 --iteration 30000
pip install open3d==0.10.0
python evaluate_dtu_mesh.py -m outputs/dtu_new/scan106 --iteration 30000 --DTU /mnt/workfiles/datasets/dtu/SampleSet/MVS_Data

python train_sdf_depth.py -s /mnt/workfiles/datasets/dtu/scan110 -m outputs/dtu_new/scan110 -r 2 --eval --save_ckpt --use_decoupled_appearance --lambda_distortion 1000 --lambda_depth 0.01 --lambda_normal 0.01
pip install --upgrade open3d
python extract_mesh_tsdf.py -m outputs/dtu_new/scan110 --iteration 30000
pip install open3d==0.10.0
python evaluate_dtu_mesh.py -m outputs/dtu_new/scan110 --iteration 30000 --DTU /mnt/workfiles/datasets/dtu/SampleSet/MVS_Data

python train_sdf_depth.py -s /mnt/workfiles/datasets/dtu/scan114 -m outputs/dtu_new/scan114 -r 2 --eval --save_ckpt --use_decoupled_appearance --lambda_distortion 1000 --lambda_depth 0.01 --lambda_normal 0.01
pip install --upgrade open3d
python extract_mesh_tsdf.py -m outputs/dtu_new/scan114 --iteration 30000
pip install open3d==0.10.0
python evaluate_dtu_mesh.py -m outputs/dtu_new/scan114 --iteration 30000 --DTU /mnt/workfiles/datasets/dtu/SampleSet/MVS_Data

python train_sdf_depth.py -s /mnt/workfiles/datasets/dtu/scan118 -m outputs/dtu_new/scan118 -r 2 --eval --save_ckpt --use_decoupled_appearance --lambda_distortion 1000 --lambda_depth 0.01 --lambda_normal 0.01
pip install --upgrade open3d
python extract_mesh_tsdf.py -m outputs/dtu_new/scan118 --iteration 30000
pip install open3d==0.10.0
python evaluate_dtu_mesh.py -m outputs/dtu_new/scan118 --iteration 30000 --DTU /mnt/workfiles/datasets/dtu/SampleSet/MVS_Data

python train_sdf_depth.py -s /mnt/workfiles/datasets/dtu/scan122 -m outputs/dtu_new/scan122 -r 2 --eval --save_ckpt --use_decoupled_appearance --lambda_distortion 1000 --lambda_depth 0.01 --lambda_normal 0.01
pip install --upgrade open3d
python extract_mesh_tsdf.py -m outputs/dtu_new/scan122 --iteration 30000
pip install open3d==0.10.0
python evaluate_dtu_mesh.py -m outputs/dtu_new/scan122 --iteration 30000 --DTU /mnt/workfiles/datasets/dtu/SampleSet/MVS_Data

