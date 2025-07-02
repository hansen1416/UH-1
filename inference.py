import os
import clip
import torch
import numpy as np
import models.tokenizer as tokenizer
import models.transformer as trans
import option

prompt_list = ["Jogging"]  # TODO
root_path = os.path.join(os.path.expanduser("~"), "checkpoints")

args = option.get_args_parser()
args.resume_tokenizer = f"{root_path}/UH-1/UH1_Action_Tokenizer.pth"
args.resume_trans = f"{root_path}/UH-1/UH1_Transformer.pth"
mean = torch.from_numpy(np.load(f"{root_path}/UH-1/Mean.npy")).cuda()
std = torch.from_numpy(np.load(f"{root_path}/UH-1/Std.npy")).cuda()

## load clip model and datasets
print("loading clip")
clip_model, clip_preprocess = clip.load(
    "ViT-B/32", device=torch.device("cuda"), jit=False
)
clip.model.convert_weights(clip_model)
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False
print("finish clip loading")

action_tokenizer = tokenizer.HumanoidVQVAE(
    args.nb_code,
    args.code_dim,
    args.output_emb_width,
    args.down_t,
    args.stride_t,
    args.width,
    args.depth,
    args.dilation_growth_rate,
)


trans_encoder = trans.UH1_Transformer(
    num_vq=args.nb_code,
    embed_dim=args.embed_dim_trans,
    clip_dim=args.clip_dim,
    block_size=args.block_size,
    num_layers=args.num_layers,
    n_head=args.n_head_trans,
    drop_out_rate=args.drop_out_rate,
    fc_rate=args.ff_rate,
)


print("loading action tokenizer checkpoint from {}".format(args.resume_tokenizer))
ckpt = torch.load(args.resume_tokenizer, map_location="cpu", weights_only=True)
action_tokenizer.load_state_dict(ckpt["net"], strict=True)
action_tokenizer.eval()
action_tokenizer.cuda()

print("loading transformer checkpoint from {}".format(args.resume_trans))
ckpt = torch.load(args.resume_trans, map_location="cpu", weights_only=True)
trans_encoder.load_state_dict(ckpt["trans"], strict=True)
trans_encoder.eval()
trans_encoder.cuda()

for clip_text in prompt_list:
    save_npy = []
    text = clip.tokenize([clip_text], truncate=True).cuda()
    feat_clip_text = clip_model.encode_text(text).float()
    index_motion = trans_encoder.sample(feat_clip_text[0:1], False)
    pred_pose = action_tokenizer.forward_decoder(index_motion) * std + mean
    save_npy.append(pred_pose.detach().cpu().numpy()[0])

    np.save(f'output/{clip_text.replace(" ", "_")}.npy', np.array(save_npy))
