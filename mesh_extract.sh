# python extract_mesh_gsdf.py -m outputs/tnt/Barn --iteration 30000
# python eval_tnt/run.py --dataset-dir eval_tnt/TrainingSet/Barn --traj-path eval_tnt/TrainingSet/Barn/Barn_COLMAP_SfM.log --ply-path outputs/tnt/Barn/test/ours_30000/fusion/mesh_binary_search_7.ply

# python extract_mesh_gsdf.py -m outputs/tnt/Caterpillar --iteration 30000
# # python extract_mesh_mc.py -m outputs/tnt/Caterpillar --iteration 30000
# python eval_tnt/run.py --dataset-dir eval_tnt/TrainingSet/Caterpillar --traj-path eval_tnt/TrainingSet/Caterpillar/Caterpillar_COLMAP_SfM.log --ply-path outputs/tnt/Caterpillar/test/ours_30000/fusion/mesh_binary_search_7.ply

# python extract_mesh_gsdf.py -m outputs/tnt/Courthouse --iteration 30000
# # python extract_mesh_mc.py -m outputs/tnt/Courthouse --iteration 30000
# python eval_tnt/run.py --dataset-dir eval_tnt/TrainingSet/Courthouse --traj-path eval_tnt/TrainingSet/Courthouse/Courthouse_COLMAP_SfM.log --ply-path outputs/tnt/Courthouse/test/ours_30000/fusion/mesh_binary_search_7.ply

# python extract_mesh_gsdf.py -m outputs/tnt/Ignatius --iteration 30000
# # python extract_mesh_mc.py -m outputs/tnt/Ignatius --iteration 30000
# python eval_tnt/run.py --dataset-dir eval_tnt/TrainingSet/Ignatius --traj-path eval_tnt/TrainingSet/Ignatius/Ignatius_COLMAP_SfM.log --ply-path outputs/tnt/Ignatius/test/ours_30000/fusion/mesh_binary_search_7.ply

# python extract_mesh_gsdf.py -m outputs/tnt/Meetingroom --iteration 30000
# python extract_mesh_mc.py -m outputs/tnt/Meetingroom --iteration 30000
# python eval_tnt/run.py --dataset-dir eval_tnt/TrainingSet/Meetingroom --traj-path eval_tnt/TrainingSet/Meetingroom/Meetingroom_COLMAP_SfM.log --ply-path outputs/tnt/Meetingroom/test/ours_30000/fusion/mesh_binary_search_7.ply

# python extract_mesh_gsdf.py -m outputs/tnt/Truck --iteration 30000
# python extract_mesh_mc.py -m outputs/tnt/Truck --iteration 30000
# python eval_tnt/run.py --dataset-dir eval_tnt/TrainingSet/Truck --traj-path eval_tnt/TrainingSet/Truck/Truck_COLMAP_SfM.log --ply-path outputs/tnt/Truck/test/ours_30000/fusion/mesh_binary_search_7.ply

# python extract_mesh_gsdf.py -m outputs/360/garden --iteration 30000
# python extract_mesh_gsdf.py -m outputs/360/bicycle --iteration 30000
# python extract_mesh_gsdf.py -m outputs/360/bonsai --iteration 30000
# python extract_mesh_gsdf.py -m outputs/360/counter --iteration 30000
# python extract_mesh_gsdf.py -m outputs/360/stump --iteration 30000
# python extract_mesh_gsdf.py -m outputs/360/kitchen --iteration 30000
# python extract_mesh_gsdf.py -m outputs/360/room --iteration 30000

# python extract_mesh_gsdf.py -m outputs/dtu/scan24 --iteration 30000
# python extract_mesh_gsdf.py -m outputs/dtu/scan37 --iteration 30000
# python extract_mesh_gsdf.py -m outputs/dtu/scan40 --iteration 30000
# python extract_mesh_gsdf.py -m outputs/dtu/scan55 --iteration 30000
# python extract_mesh_gsdf.py -m outputs/dtu/scan63 --iteration 30000
# python extract_mesh_gsdf.py -m outputs/dtu/scan65 --iteration 30000
# python extract_mesh_gsdf.py -m outputs/dtu/scan69 --iteration 30000
# python extract_mesh_gsdf.py -m outputs/dtu/scan83 --iteration 30000
# python extract_mesh_gsdf.py -m outputs/dtu/scan97 --iteration 30000
# python extract_mesh_gsdf.py -m outputs/dtu/scan105 --iteration 30000
# python extract_mesh_gsdf.py -m outputs/dtu/scan106 --iteration 30000
# python extract_mesh_gsdf.py -m outputs/dtu/scan110 --iteration 30000
# python extract_mesh_gsdf.py -m outputs/dtu/scan114 --iteration 30000
# python extract_mesh_gsdf.py -m outputs/dtu/scan118 --iteration 30000
# python extract_mesh_gsdf.py -m outputs/dtu/scan122 --iteration 30000

# python train_sdf_depth.py -s /mnt/workfiles/datasets/dtu/scan24 -m outputs/dtu/scan24 -r 2 --eval --save_ckpt --lambda_distortion 1000
# python train_sdf_depth.py -s /mnt/workfiles/datasets/dtu/scan37 -m outputs/dtu/scan37 -r 2 --eval --save_ckpt --lambda_distortion 1000
# python train_sdf_depth.py -s /mnt/workfiles/datasets/dtu/scan40 -m outputs/dtu/scan40 -r 2 --eval --save_ckpt --lambda_distortion 1000
# python train_sdf_depth.py -s /mnt/workfiles/datasets/dtu/scan55 -m outputs/dtu/scan55 -r 2 --eval --save_ckpt --lambda_distortion 1000
# python train_sdf_depth.py -s /mnt/workfiles/datasets/dtu/scan63 -m outputs/dtu/scan63 -r 2 --eval --save_ckpt --lambda_distortion 1000
# python train_sdf_depth.py -s /mnt/workfiles/datasets/dtu/scan65 -m outputs/dtu/scan65 -r 2 --eval --save_ckpt --lambda_distortion 1000
# python train_sdf_depth.py -s /mnt/workfiles/datasets/dtu/scan69 -m outputs/dtu/scan69 -r 2 --eval --save_ckpt --lambda_distortion 1000
# python train_sdf_depth.py -s /mnt/workfiles/datasets/dtu/scan83 -m outputs/dtu/scan83 -r 2 --eval --save_ckpt --lambda_distortion 1000
# python train_sdf_depth.py -s /mnt/workfiles/datasets/dtu/scan97 -m outputs/dtu/scan97 -r 2 --eval --save_ckpt --lambda_distortion 1000
# python train_sdf_depth.py -s /mnt/workfiles/datasets/dtu/scan105 -m outputs/dtu/scan105 -r 2 --eval --save_ckpt --lambda_distortion 1000
# python train_sdf_depth.py -s /mnt/workfiles/datasets/dtu/scan106 -m outputs/dtu/scan106 -r 2 --eval --save_ckpt --lambda_distortion 1000
# python train_sdf_depth.py -s /mnt/workfiles/datasets/dtu/scan110 -m outputs/dtu/scan110 -r 2 --eval --save_ckpt --lambda_distortion 1000
# python train_sdf_depth.py -s /mnt/workfiles/datasets/dtu/scan114 -m outputs/dtu/scan114 -r 2 --eval --save_ckpt --lambda_distortion 1000
# python train_sdf_depth.py -s /mnt/workfiles/datasets/dtu/scan118 -m outputs/dtu/scan118 -r 2 --eval --save_ckpt --lambda_distortion 1000
# python train_sdf_depth.py -s /mnt/workfiles/datasets/dtu/scan122 -m outputs/dtu/scan122 -r 2 --eval --save_ckpt --lambda_distortion 1000

# python render.py -m outputs/360/garden --data_device cpu --skip_train
# python metrics.py -m outputs/360/garden -r 4
# python render.py -m outputs/360/bicycle --data_device cpu --skip_train
# python metrics.py -m outputs/360/bicycle -r 4
# python render.py -m outputs/360/bonsai --data_device cpu --skip_train
# python metrics.py -m outputs/360/bonsai -r 2
# python render.py -m outputs/360/counter --data_device cpu --skip_train
# python metrics.py -m outputs/360/counter -r 2
# python render.py -m outputs/360/stump --data_device cpu --skip_train
# python metrics.py -m outputs/360/stump -r 4
# python render.py -m outputs/360/kitchen --data_device cpu --skip_train
# python metrics.py -m outputs/360/kitchen -r 2
# python render.py -m outputs/360/room --data_device cpu --skip_train
# python metrics.py -m outputs/360/room -r 2
