# # Generate activation smoothing scale 
# python generate_act_scale.py --resume checkpoints/vim_s_midclstok_80p5acc.pth --model vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 --data-path /share/public/imagenet --batch-size 256

# torchrun --nproc_per_node 1 generate_act_scale.py --resume checkpoints/vim_s_midclstok_80p5acc.pth --model vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 --data-path /share/public/imagenet --batch-size 256


# Joint Learning of Smoothing Scale and Step size (JLSS)
# to obtain the 4bit model
torchrun --nproc_per_node 1 quant.py --eval --resume checkpoints/vim_s_midclstok_80p5acc.pth --model vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 --data-path /share/public/imagenet --act_scales ./smoothing_s.pt --batch-size 256 --qmode ptq4vm --train-batch 256 --n-lva 16 --n-lvw 16 --alpha 0.5 --epochs 100 --lr-a 5e-4 --lr-w 5e-4 --lr-s 1e-2 --save-ckpt
python quant.py --eval --resume checkpoints/vim_s_midclstok_80p5acc.pth --model vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 --data-path /share/public/imagenet --act_scales ./smoothing_s.pt --batch-size 256 --qmode ptq4vm --train-batch 256 --n-lva 16 --n-lvw 16 --alpha 0.5 --epochs 100 --lr-a 5e-4 --lr-w 5e-4 --lr-s 1e-2 --save-ckpt

# to obtain the 6bit model
torchrun --nproc_per_node 1 quant.py --eval --resume checkpoints/vim_s_midclstok_80p5acc.pth --model vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 --data-path /share/public/imagenet --act_scales ./smoothing_s.pt --batch-size 256 --qmode ptq4vm --train-batch 256 --n-lva 64 --n-lvw 64 --alpha 0.5 --epochs 100 --lr-a 5e-4 --lr-w 5e-4 --lr-s 1e-2 --save-ckpt
python quant.py --eval --resume checkpoints/vim_s_midclstok_80p5acc.pth --model vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 --data-path /share/public/imagenet --act_scales ./smoothing_s.pt --batch-size 256 --qmode ptq4vm --train-batch 256 --n-lva 64 --n-lvw 64 --alpha 0.5 --epochs 100 --lr-a 5e-4 --lr-w 5e-4 --lr-s 1e-2 --save-ckpt

# to obtain the 8bit model
torchrun --nproc_per_node 1 quant.py --eval --resume checkpoints/vim_s_midclstok_80p5acc.pth --model vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 --data-path /share/public/imagenet --act_scales ./smoothing_s.pt --batch-size 256 --qmode ptq4vm --train-batch 256 --n-lva 256 --n-lvw 256 --alpha 0.5 --epochs 100 --lr-a 5e-4 --lr-w 5e-4 --lr-s 1e-2 --save-ckpt

python quant.py --eval --resume checkpoints/vim_s_midclstok_80p5acc.pth --model vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 --data-path /share/public/imagenet --act_scales ./smoothing_s.pt --batch-size 256 --qmode ptq4vm --train-batch 256 --n-lva 256 --n-lvw 256 --alpha 0.5 --epochs 100 --lr-a 5e-4 --lr-w 5e-4 --lr-s 1e-2 --save-ckpt

# test the 4bit model
python quant.py --eval --resume output/vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2_W4A4.pt --model vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 --data-path /share/public/imagenet --act_scales ./smoothing_s.pt --batch-size 64 --qmode '' --train-batch 64 --n-lva 16 --n-lvw 16

# test the 6bit model
python quant.py --eval --resume output/vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2_W6A6.pt --model vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 --data-path /share/public/imagenet --act_scales ./smoothing_s.pt --batch-size 64 --qmode '' --train-batch 64 --n-lva 64 --n-lvw 64