ulimit -n 4096
# tnt
python train_sdf_depth.py -s /mnt/workfiles/datasets/tnt/TrainingSet/Barn -m outputs/tnt_1122/Barn -r 2 --eval --save_ckpt --use_decoupled_appearance #--ckpt_pth outputs/tnt_1122/Barn/ckpt/ckpt_20000.pth
python extract_mesh_gsdf.py -s /mnt/workfiles/datasets/tnt/TrainingSet/Barn -m outputs/tnt_1122/Barn -r  --iteration 30000
python eval_tnt/run.py --dataset-dir eval_tnt/TrainingSet/Barn --traj-path /mnt/workfiles/datasets/tnt/TrainingSet/Barn/Barn_COLMAP_SfM.log --ply-path outputs/tnt_1122/Barn/test/ours_30000/fusion/mesh_binary_search_7.ply
python extract_mesh_mc.py -s /mnt/workfiles/datasets/tnt/TrainingSet/Barn -m outputs/tnt_1122/Barn --mesh_res 1024 --iteration 30000 --unbounded

python train_sdf_depth.py -s /mnt/workfiles/datasets/tnt/TrainingSet/Caterpillar -m outputs/tnt_1122/Caterpillar -r 2 --eval --save_ckpt --use_decoupled_appearance
python extract_mesh_gsdf.py -s /mnt/workfiles/datasets/tnt/TrainingSet/Caterpillar -m outputs/tnt_1122/Caterpillar -r 2 --iteration 30000
python eval_tnt/run.py --dataset-dir eval_tnt/TrainingSet/Caterpillar --traj-path /mnt/workfiles/datasets/tnt/TrainingSet/Caterpillar/Caterpillar_COLMAP_SfM.log --ply-path outputs/tnt_1122/Caterpillar/test/ours_30000/fusion/mesh_binary_search_7.ply
python extract_mesh_mc.py -s /mnt/workfiles/datasets/tnt/TrainingSet/Caterpillar -m outputs/tnt_1122/Caterpillar --mesh_res 2048 --iteration 30000 --unbounded

python train_sdf_depth.py -s /mnt/workfiles/datasets/tnt/TrainingSet/Courthouse -m outputs/tnt_1122/Courthouse -r 2 --eval --save_ckpt --use_decoupled_appearance
python extract_mesh_gsdf.py -s /mnt/workfiles/datasets/tnt/TrainingSet/Courthouse -m outputs/tnt_1122/Courthouse -r 2 --iteration 30000
python eval_tnt/run.py --dataset-dir eval_tnt/TrainingSet/Courthouse --traj-path /mnt/workfiles/datasets/tnt/TrainingSet/Courthouse/Courthouse_COLMAP_SfM.log --ply-path outputs/tnt_1122/Courthouse/test/ours_30000/fusion/mesh_binary_search_7.ply
python extract_mesh_mc.py -s /mnt/workfiles/datasets/tnt/TrainingSet/Courthouse -m outputs/tnt_1122/Courthouse --mesh_res 2048 --iteration 30000 --unbounded

python train_sdf_depth.py -s /mnt/workfiles/datasets/tnt/TrainingSet/Ignatius -m outputs/tnt_1122/Ignatius -r 2 --eval --save_ckpt --use_decoupled_appearance
python extract_mesh_gsdf.py -s /mnt/workfiles/datasets/tnt/TrainingSet/Ignatius -m outputs/tnt_1122/Ignatius -r 2 --iteration 30000
python eval_tnt/run.py --dataset-dir eval_tnt/TrainingSet/Ignatius --traj-path /mnt/workfiles/datasets/tnt/TrainingSet/Ignatius/Ignatius_COLMAP_SfM.log --ply-path outputs/tnt_1122/Ignatius/test/ours_30000/fusion/mesh_binary_search_7.ply
python extract_mesh_mc.py -s /mnt/workfiles/datasets/tnt/TrainingSet/Ignatius -m outputs/tnt_1122/Ignatius --mesh_res 2048 --iteration 30000 --unbounded

python train_sdf_depth.py -s /mnt/workfiles/datasets/tnt/TrainingSet/Meetingroom -m outputs/tnt_1122/Meetingroom -r 2 --eval --save_ckpt --use_decoupled_appearance
python extract_mesh_gsdf.py -s /mnt/workfiles/datasets/tnt/TrainingSet/Meetingroom -m outputs/tnt_1122/Meetingroom -r 2 --iteration 30000
python eval_tnt/run.py --dataset-dir eval_tnt/TrainingSet/Meetingroom --traj-path /mnt/workfiles/datasets/tnt/TrainingSet/Meetingroom/Meetingroom_COLMAP_SfM.log --ply-path outputs/tnt_1122/Meetingroom/test/ours_30000/fusion/mesh_binary_search_7.ply
python extract_mesh_mc.py -s /mnt/workfiles/datasets/tnt/TrainingSet/Meetingroom -m outputs/tnt_1122/Meetingroom --mesh_res 2048 --iteration 30000 --unbounded

python train_sdf_depth.py -s /mnt/workfiles/datasets/tnt/TrainingSet/Truck -m outputs/tnt_1122/Truck -r 2 --eval --save_ckpt --use_decoupled_appearance
python extract_mesh_gsdf.py -s /mnt/workfiles/datasets/tnt/TrainingSet/Truck -m outputs/tnt_1122/Truck -r 2 --iteration 30000
python eval_tnt/run.py --dataset-dir eval_tnt/TrainingSet/Truck --traj-path /mnt/workfiles/datasets/tnt/TrainingSet/Truck/Truck_COLMAP_SfM.log --ply-path outputs/tnt_1122/Truck/test/ours_30000/fusion/mesh_binary_search_7.ply
python extract_mesh_mc.py -s /mnt/workfiles/datasets/tnt/TrainingSet/Truck -m outputs/tnt_1122/Truck --mesh_res 2048 --iteration 30000 --unbounded

# # MIP 360
# python train_sdf_depth.py -s /mnt/workfiles/datasets/360/garden -m outputs/360_new/garden -i images_4 --eval --save_ckpt --lambda_depth 0.01 --lambda_normal 0.01
# python render.py -m outputs/360_new/garden --data_device cpu --skip_train
# python metrics.py -m outputs/360_new/garden
# python extract_mesh_gsdf.py -m outputs/360_new/garden --iteration 30000

# python train_sdf_depth.py -s /mnt/workfiles/datasets/360/bicycle -m outputs/360_new/bicycle -i images_4 --eval --save_ckpt --lambda_depth 0.01 --lambda_normal 0.01 #--ckpt_pth outputs/360_new/bicycle/ckpt/ckpt_10000.pth
# python render.py -m outputs/360_new/bicycle --data_device cpu --skip_train
# python metrics.py -m outputs/360_new/bicycle
# python extract_mesh_gsdf.py -m outputs/360_new/bicycle --iteration 30000

# python train_sdf_depth.py -s /mnt/workfiles/datasets/360/stump -m outputs/360_new/stump -i images_4 --eval --save_ckpt --lambda_depth 0.01 --lambda_normal 0.01
# python render.py -m outputs/360_new/stump --data_device cpu --skip_train
# python metrics.py -m outputs/360_new/stump
# python extract_mesh_gsdf.py -m outputs/360_new/stump --iteration 30000

# python train_sdf_depth.py -s /mnt/workfiles/datasets/360/room -m outputs/360_new/room -i images_2 --eval --save_ckpt --lambda_depth 0.01 --lambda_normal 0.01
# python render.py -m outputs/360_new/room --data_device cpu --skip_train
# python metrics.py -m outputs/360_new/room
# python extract_mesh_gsdf.py -m outputs/360_new/room --iteration 30000

# python train_sdf_depth.py -s /mnt/workfiles/datasets/360/bonsai -m outputs/360_new/bonsai -i images_2 --eval --save_ckpt --lambda_depth 0.01 --lambda_normal 0.01
# python render.py -m outputs/360_new/bonsai --data_device cpu --skip_train
# python metrics.py -m outputs/360_new/bonsai
# python extract_mesh_gsdf.py -m outputs/360_new/bonsai --iteration 30000

# python train_sdf_depth.py -s /mnt/workfiles/datasets/360/counter -m outputs/360_new/counter -i images_2 --eval --save_ckpt --lambda_depth 0.01 --lambda_normal 0.01
# python render.py -m outputs/360_new/counter --data_device cpu --skip_train
# python metrics.py -m outputs/360_new/counter
# python extract_mesh_gsdf.py -m outputs/360_new/counter --iteration 30000

# python train_sdf_depth.py -s /mnt/workfiles/datasets/360/kitchen -m outputs/360_new/kitchen -i images_2 --eval --save_ckpt --lambda_depth 0.01 --lambda_normal 0.01
# python render.py -m outputs/360_new/kitchen --data_device cpu --skip_train
# python metrics.py -m outputs/360_new/kitchen
# python extract_mesh_gsdf.py -m outputs/360_new/kitchen --iteration 30000

# python train_sdf_depth.py -s /mnt/workfiles/datasets/360/flowers -m outputs/360_new/flowers -i images_4 --eval --save_ckpt --lambda_depth 0.01 --lambda_normal 0.01
# python render.py -m outputs/360_new/flowers --data_device cpu --skip_train
# python metrics.py -m outputs/360_new/flowers
# python extract_mesh_gsdf.py -m outputs/360_1113/flowers --iteration 30000

# python train_sdf_depth.py -s /mnt/workfiles/datasets/360/treehill -m outputs/360_new/treehill -i images_4 --eval --save_ckpt --lambda_depth 0.01 --lambda_normal 0.01
# python render.py -m outputs/360_new/treehill --data_device cpu --skip_train
# python metrics.py -m outputs/360_new/treehill
# python extract_mesh_gsdf.py -m outputs/360_1113/treehill --iteration 30000

# DTU
# python train_sdf_depth.py -s /mnt/workfiles/datasets/dtu/scan24 -m outputs/dtu_1010/scan24 -r 2 --eval --save_ckpt --lambda_distortion 1000 --lambda_depth 0.01 
# pip install --upgrade open3d
# python extract_mesh_tsdf.py -m outputs/dtu_1010/scan24 --iteration 30000
# pip install open3d==0.10.0
# python evaluate_dtu_mesh.py -m outputs/dtu_1010/scan24 --iteration 30000 --DTU /mnt/workfiles/datasets/dtu/SampleSet/MVS_Data

# python train_sdf_depth.py -s /mnt/workfiles/datasets/dtu/scan37 -m outputs/dtu_1010/scan37 -r 2 --eval --save_ckpt --lambda_distortion 1000 --lambda_depth 0.01 
# pip install --upgrade open3d
# python extract_mesh_tsdf.py -m outputs/dtu_1010/scan37 --iteration 30000
# pip install open3d==0.10.0
# python evaluate_dtu_mesh.py -m outputs/dtu_1010/scan37 --iteration 30000 --DTU /mnt/workfiles/datasets/dtu/SampleSet/MVS_Data

# python train_sdf_depth.py -s /mnt/workfiles/datasets/dtu/scan40 -m outputs/dtu_1010/scan40 -r 2 --eval --save_ckpt --lambda_distortion 1000 --lambda_depth 0.01 
# pip install --upgrade open3d
# python extract_mesh_tsdf.py -m outputs/dtu_1010/scan40 --iteration 30000
# pip install open3d==0.10.0
# python evaluate_dtu_mesh.py -m outputs/dtu_1010/scan40 --iteration 30000 --DTU /mnt/workfiles/datasets/dtu/SampleSet/MVS_Data

# python train_sdf_depth.py -s /mnt/workfiles/datasets/dtu/scan55 -m outputs/dtu_1010/scan55 -r 2 --eval --save_ckpt --lambda_distortion 1000 --lambda_depth 0.01 
# pip install --upgrade open3d
# python extract_mesh_tsdf.py -m outputs/dtu_1010/scan55 --iteration 30000
# pip install open3d==0.10.0
# python evaluate_dtu_mesh.py -m outputs/dtu_1010/scan55 --iteration 30000 --DTU /mnt/workfiles/datasets/dtu/SampleSet/MVS_Data

# python train_sdf_depth.py -s /mnt/workfiles/datasets/dtu/scan63 -m outputs/dtu_1010/scan63 -r 2 --eval --save_ckpt --lambda_distortion 1000 --lambda_depth 0.01  #--ckpt_pth outputs/dtu_1010/scan63/ckpt/ckpt_20000.pth
# pip install --upgrade open3d
# python extract_mesh_tsdf.py -m outputs/dtu_1010/scan63 --iteration 30000
# pip install open3d==0.10.0
# python evaluate_dtu_mesh.py -m outputs/dtu_1010/scan63 --iteration 30000 --DTU /mnt/workfiles/datasets/dtu/SampleSet/MVS_Data

# python train_sdf_depth.py -s /mnt/workfiles/datasets/dtu/scan65 -m outputs/dtu_1010/scan65 -r 2 --eval --save_ckpt --lambda_distortion 1000 --lambda_depth 0.01 
# pip install --upgrade open3d
# python extract_mesh_tsdf.py -m outputs/dtu_1010/scan65 --iteration 30000
# pip install open3d==0.10.0
# python evaluate_dtu_mesh.py -m outputs/dtu_1010/scan65 --iteration 30000 --DTU /mnt/workfiles/datasets/dtu/SampleSet/MVS_Data

# python train_sdf_depth.py -s /mnt/workfiles/datasets/dtu/scan69 -m outputs/dtu_1010/scan69 -r 2 --eval --save_ckpt --lambda_distortion 1000 --lambda_depth 0.01 
# pip install --upgrade open3d
# python extract_mesh_tsdf.py -m outputs/dtu_1010/scan69 --iteration 30000
# pip install open3d==0.10.0
# python evaluate_dtu_mesh.py -m outputs/dtu_1010/scan69 --iteration 30000 --DTU /mnt/workfiles/datasets/dtu/SampleSet/MVS_Data

# python train_sdf_depth.py -s /mnt/workfiles/datasets/dtu/scan83 -m outputs/dtu_1010/scan83 -r 2 --eval --save_ckpt --lambda_distortion 1000 --lambda_depth 0.01 
# pip install --upgrade open3d
# python extract_mesh_tsdf.py -m outputs/dtu_1010/scan83 --iteration 30000
# pip install open3d==0.10.0
# python evaluate_dtu_mesh.py -m outputs/dtu_1010/scan83 --iteration 30000 --DTU /mnt/workfiles/datasets/dtu/SampleSet/MVS_Data

# python train_sdf_depth.py -s /mnt/workfiles/datasets/dtu/scan97 -m outputs/dtu_1010/scan97 -r 2 --eval --save_ckpt --lambda_distortion 1000 --lambda_depth 0.01 
# pip install --upgrade open3d
# python extract_mesh_tsdf.py -m outputs/dtu_1010/scan97 --iteration 30000
# pip install open3d==0.10.0
# python evaluate_dtu_mesh.py -m outputs/dtu_1010/scan97 --iteration 30000 --DTU /mnt/workfiles/datasets/dtu/SampleSet/MVS_Data

# python train_sdf_depth.py -s /mnt/workfiles/datasets/dtu/scan105 -m outputs/dtu_1010/scan105 -r 2 --eval --save_ckpt --lambda_distortion 1000 --lambda_depth 0.01 
# pip install --upgrade open3d
# python extract_mesh_tsdf.py -m outputs/dtu_1010/scan105 --iteration 30000
# pip install open3d==0.10.0
# python evaluate_dtu_mesh.py -m outputs/dtu_1010/scan105 --iteration 30000 --DTU /mnt/workfiles/datasets/dtu/SampleSet/MVS_Data

# python train_sdf_depth.py -s /mnt/workfiles/datasets/dtu/scan106 -m outputs/dtu_1010/scan106 -r 2 --eval --save_ckpt --lambda_distortion 1000 --lambda_depth 0.01 
# pip install --upgrade open3d
# python extract_mesh_tsdf.py -m outputs/dtu_1010/scan106 --iteration 30000
# pip install open3d==0.10.0
# python evaluate_dtu_mesh.py -m outputs/dtu_1010/scan106 --iteration 30000 --DTU /mnt/workfiles/datasets/dtu/SampleSet/MVS_Data

# python train_sdf_depth.py -s /mnt/workfiles/datasets/dtu/scan110 -m outputs/dtu_1010/scan110 -r 2 --eval --save_ckpt --lambda_distortion 1000 --lambda_depth 0.01 
# pip install --upgrade open3d
# python extract_mesh_tsdf.py -m outputs/dtu_1010/scan110 --iteration 30000
# pip install open3d==0.10.0
# python evaluate_dtu_mesh.py -m outputs/dtu_1010/scan110 --iteration 30000 --DTU /mnt/workfiles/datasets/dtu/SampleSet/MVS_Data

# python train_sdf_depth.py -s /mnt/workfiles/datasets/dtu/scan114 -m outputs/dtu_1010/scan114 -r 2 --eval --save_ckpt --lambda_distortion 1000 --lambda_depth 0.01 
# pip install --upgrade open3d
# python extract_mesh_tsdf.py -m outputs/dtu_1010/scan114 --iteration 30000
# pip install open3d==0.10.0
# python evaluate_dtu_mesh.py -m outputs/dtu_1010/scan114 --iteration 30000 --DTU /mnt/workfiles/datasets/dtu/SampleSet/MVS_Data

# python train_sdf_depth.py -s /mnt/workfiles/datasets/dtu/scan118 -m outputs/dtu_1010/scan118 -r 2 --eval --save_ckpt --lambda_distortion 1000 --lambda_depth 0.01 
# pip install --upgrade open3d
# python extract_mesh_tsdf.py -m outputs/dtu_1010/scan118 --iteration 30000
# pip install open3d==0.10.0
# python evaluate_dtu_mesh.py -m outputs/dtu_1010/scan118 --iteration 30000 --DTU /mnt/workfiles/datasets/dtu/SampleSet/MVS_Data

# python train_sdf_depth.py -s /mnt/workfiles/datasets/dtu/scan122 -m outputs/dtu_1010/scan122 -r 2 --eval --save_ckpt --lambda_distortion 1000 --lambda_depth 0.01 
# pip install --upgrade open3d
# python extract_mesh_tsdf.py -m outputs/dtu_1010/scan122 --iteration 30000
# pip install open3d==0.10.0
# python evaluate_dtu_mesh.py -m outputs/dtu_1010/scan122 --iteration 30000 --DTU /mnt/workfiles/datasets/dtu/SampleSet/MVS_Data
