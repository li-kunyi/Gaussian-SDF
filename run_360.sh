ulimit -n 4096

# TNT
# python train_sdf_depth.py -s /mnt/workfiles/datasets/tnt/TrainingSet/Barn -m outputs/tnt_1105/Barn -r 2 --eval --save_ckpt --use_decoupled_appearance #--ckpt_pth outputs/tnt_1105/Barn/ckpt/ckpt_20000.pth
# python extract_mesh_gsdf.py -m outputs/tnt_1105/Barn --iteration 30000
# # pip install --upgrade open3d
# # python extract_mesh_tsdf.py -m outputs/tnt_1105/Barn --iteration 30000
# # pip install open3d==0.10.0
# python eval_tnt/run.py --dataset-dir eval_tnt/TrainingSet/Barn --traj-path /mnt/workfiles/datasets/tnt/TrainingSet/Barn/Barn_COLMAP_SfM.log --ply-path outputs/tnt_1105/Barn/test/ours_30000/fusion/mesh_binary_search_7.ply

# python train_sdf_depth.py -s /mnt/workfiles/datasets/tnt/TrainingSet/Caterpillar -m outputs/tnt_1105/Caterpillar -r 2 --eval --save_ckpt --use_decoupled_appearance
# python extract_mesh_gsdf.py -m outputs/tnt_1105/Caterpillar --iteration 30000
# # pip install --upgrade open3d
# # python extract_mesh_tsdf.py -m outputs/tnt_1105/Caterpillar --iteration 30000
# # pip install open3d==0.10.0
# python eval_tnt/run.py --dataset-dir eval_tnt/TrainingSet/Caterpillar --traj-path /mnt/workfiles/datasets/tnt/TrainingSet/Caterpillar/Caterpillar_COLMAP_SfM.log --ply-path outputs/tnt_1105/Caterpillar/test/ours_30000/fusion/mesh_binary_search_7.ply

# python train_sdf_depth.py -s /mnt/workfiles/datasets/tnt/TrainingSet/Courthouse -m outputs/tnt_1105/Courthouse -r 2 --eval --save_ckpt --use_decoupled_appearance
# python extract_mesh_gsdf.py -m outputs/tnt_1105/Courthouse --iteration 30000
# # pip install --upgrade open3d
# # python extract_mesh_tsdf.py -m outputs/tnt_1105/Courthouse --iteration 30000
# # pip install open3d==0.10.0
# python eval_tnt/run.py --dataset-dir eval_tnt/TrainingSet/Courthouse --traj-path /mnt/workfiles/datasets/tnt/TrainingSet/Courthouse/Courthouse_COLMAP_SfM.log --ply-path outputs/tnt_1105/Courthouse/test/ours_30000/fusion/mesh_binary_search_7.ply

# python train_sdf_depth.py -s /mnt/workfiles/datasets/tnt/TrainingSet/Ignatius -m outputs/tnt_1105/Ignatius -r 2 --eval --save_ckpt --use_decoupled_appearance
# python extract_mesh_gsdf.py -m outputs/tnt_1105/Ignatius --iteration 30000
# # pip install --upgrade open3d
# # python extract_mesh_tsdf.py -m outputs/tnt_1105/Ignatius --iteration 30000
# # pip install open3d==0.10.0
# python eval_tnt/run.py --dataset-dir eval_tnt/TrainingSet/Ignatius --traj-path /mnt/workfiles/datasets/tnt/TrainingSet/Ignatius/Ignatius_COLMAP_SfM.log --ply-path outputs/tnt_1105/Ignatius/test/ours_30000/fusion/mesh_binary_search_7.ply

# python train_sdf_depth.py -s /mnt/workfiles/datasets/tnt/TrainingSet/Meetingroom -m outputs/tnt_1105/Meetingroom -r 2 --eval --save_ckpt --use_decoupled_appearance
# python extract_mesh_gsdf.py -m outputs/tnt_1105/Meetingroom --iteration 30000
# # pip install --upgrade open3d
# # python extract_mesh_tsdf.py -m outputs/tnt_tnt_11051011/Meetingroom --iteration 30000
# # pip install open3d==0.10.0
# python eval_tnt/run.py --dataset-dir eval_tnt/TrainingSet/Meetingroom --traj-path /mnt/workfiles/datasets/tnt/TrainingSet/Meetingroom/Meetingroom_COLMAP_SfM.log --ply-path outputs/tnt_1105/Meetingroom/test/ours_30000/fusion/mesh_binary_search_7.ply

# python train_sdf_depth.py -s /mnt/workfiles/datasets/tnt/TrainingSet/Truck -m outputs/tnt_1105/Truck -r 2 --eval --save_ckpt --use_decoupled_appearance
# python extract_mesh_gsdf.py -m outputs/tnt_1105/Truck --iteration 30000
# # pip install --upgrade open3d
# # python extract_mesh_tsdf.py -m outputs/tnt_1105/Truck --iteration 30000
# # pip install open3d==0.10.0
# python eval_tnt/run.py --dataset-dir eval_tnt/TrainingSet/Truck --traj-path /mnt/workfiles/datasets/tnt/TrainingSet/Truck/Truck_COLMAP_SfM.log --ply-path outputs/tnt_1105/Truck/test/ours_30000/fusion/mesh_binary_search_7.ply

# MIP 360
# python train_sdf_depth.py -s /mnt/workfiles/datasets/360/garden -m outputs/360_1011/garden -r 4 --eval --save_ckpt --lambda_depth 0.01 --lambda_normal 0.01
# python render.py -m outputs/360_1011/garden --data_device cpu --skip_train
# python metrics.py -m outputs/360_1011/garden -r 4
python extract_mesh_gsdf.py -m outputs/360_1011/garden --iteration 30000

# python train_sdf_depth.py -s /mnt/workfiles/datasets/360/bicycle -m outputs/360_1011/bicycle -r 4 --eval --save_ckpt --lambda_depth 0.01 --lambda_normal 0.01
# python render.py -m outputs/360_1011/bicycle --data_device cpu --skip_train
# python metrics.py -m outputs/360_1011/bicycle -r 4
python extract_mesh_gsdf.py -m outputs/360_1011/bicycle --iteration 30000

# python train_sdf_depth.py -s /mnt/workfiles/datasets/360/stump -m outputs/360_1011/stump -r 4 --eval --save_ckpt --lambda_depth 0.01 --lambda_normal 0.01
# python render.py -m outputs/360_1011/stump --data_device cpu --skip_train
# python metrics.py -m outputs/360_1011/stump -r 4
python extract_mesh_gsdf.py -m outputs/360_1011/stump --iteration 30000

# python train_sdf_depth.py -s /mnt/workfiles/datasets/360/room -m outputs/360_1011/room -r 2 --eval --save_ckpt --lambda_depth 0.01 --lambda_normal 0.01
# python render.py -m outputs/360_1011/room --data_device cpu --skip_train
# python metrics.py -m outputs/360_1011/room -r 2
# python extract_mesh_gsdf.py -m outputs/360_1011/room --iteration 30000

# python train_sdf_depth.py -s /mnt/workfiles/datasets/360/bonsai -m outputs/360_1011/bonsai -r 2 --eval --save_ckpt --lambda_depth 0.01 --lambda_normal 0.01
# python render.py -m outputs/360_1011/bonsai --data_device cpu --skip_train
# python metrics.py -m outputs/360_1011/bonsai -r 2
# python extract_mesh_gsdf.py -m outputs/360_1011/bonsai --iteration 30000

# python train_sdf_depth.py -s /mnt/workfiles/datasets/360/counter -m outputs/360_1011/counter -r 2 --eval --save_ckpt --lambda_depth 0.01 --lambda_normal 0.01
# python render.py -m outputs/360_1011/counter --data_device cpu --skip_train
# python metrics.py -m outputs/360_1011/counter -r 2
# python extract_mesh_gsdf.py -m outputs/360_1011/counter --iteration 30000

# python train_sdf_depth.py -s /mnt/workfiles/datasets/360/kitchen -m outputs/360_1011/kitchen -r 2 --eval --save_ckpt --lambda_depth 0.01 --lambda_normal 0.01
# python render.py -m outputs/360_1011/kitchen --data_device cpu --skip_train
# python metrics.py -m outputs/360_1011/kitchen -r 2
# python extract_mesh_gsdf.py -m outputs/360_1011/kitchen --iteration 30000
