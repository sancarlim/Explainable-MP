import argparse
import yaml
from train_eval.trainer import Trainer
from torch.utils.tensorboard import SummaryWriter
import os
import wandb

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="Config file with dataset parameters", required=True)
parser.add_argument("-r", "--data_root", help="Root directory with data", required=True)
parser.add_argument("-d", "--data_dir", help="Directory to extract data", required=True)
parser.add_argument("-o", "--output_dir", help="Directory to save checkpoints and logs", required=True)
parser.add_argument("-n", "--num_epochs", help="Number of epochs to run training for", required=True)
parser.add_argument("-w", "--checkpoint", help="Path to pre-trained or intermediate checkpoint", required=False)
parser.add_argument('--nowandb', action='store_true', help='use this flag to DISABLE wandb logging') 
parser.add_argument('--sweep', action='store_true', help='use this flag to indicate that this is a sweep run')  
parser.add_argument('--aggregator_args.num_heads', type=int, default=16, help='number of heads for aggregator')
parser.add_argument('--aggregator_args.pre_train', type=bool, default=True)
parser.add_argument('--encoder_args.target_agent_enc_size', type=int, default=128)
parser.add_argument('--encoder_args.target_agent_emb_size', type=int, default=64)
parser.add_argument('--encoder_args.num_heads_lanes', type=int, default=1)
parser.add_argument('--encoder_args.feat_drop', type=float, default=0.)
parser.add_argument('--encoder_args.attn_drop', type=float, default=0.)
parser.add_argument('--encoder_args.num_layers', type=int, default=3)
parser.add_argument('--encoder_args.node_hgt_size', type=int, default=32)
parser.add_argument('--encoder_args.hg', type=str, default="simple")
parser.add_argument('--optim_args.scheduler_step', type=int, default=10)
parser.add_argument('--optim_args.lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=16)


args = parser.parse_args()

# Load config
with open(args.config, 'r') as yaml_file:
    cfg = yaml.safe_load(yaml_file)

# Initialize wandb loger
wandb_logger = None
if not args.nowandb:
    wandb_logger = wandb.init(job_type="training", entity='sandracl72', project='nuscenes_pgp',
                            config=cfg, sync_tensorboard=True) 
    wandb_logger.name=wandb.run.name
    if args.sweep:
        enc_args = {key.split('.')[-1]: value for key, value in vars(args).items() if 'encoder' in key.lower()}
        agg_args = {key.split('.')[-1]: value for key, value in vars(args).items() if 'aggregator' in key.lower()}
        optim_args = {key.split('.')[-1]: value for key, value in vars(args).items() if 'optim' in key.lower()}
        cfg['encoder_args'].update(enc_args)
        cfg['aggregator_args'].update(agg_args)
        cfg['optim_args'].update(optim_args)
        cfg.update({'batch_size': args.batch_size})
        cfg['encoder_args'].update({'num_heads_lanes': [enc_args['num_heads_lanes']]*enc_args['num_layers']})
        cfg['encoder_args'].update({'node_emb_size': enc_args['target_agent_emb_size']})
        cfg['encoder_args'].update({'nbr_emb_size': enc_args['target_agent_emb_size']})
        cfg['encoder_args'].update({'node_enc_size': enc_args['target_agent_enc_size']})
        cfg['encoder_args'].update({'nbr_enc_size': enc_args['target_agent_enc_size']})
        cfg['encoder_args'].update({'node_out_hgt_size': enc_args['target_agent_enc_size']})
        cfg['aggregator_args'].update({'target_agent_enc_size': enc_args['target_agent_enc_size']*2})
        cfg['aggregator_args'].update({'node_enc_size': enc_args['target_agent_enc_size']})
        cfg['aggregator_args'].update({'pi_h1_size': enc_args['target_agent_enc_size']})
        cfg['aggregator_args'].update({'pi_h2_size': enc_args['target_agent_enc_size']})
        cfg['aggregator_args'].update({'emb_size': enc_args['target_agent_enc_size']*4})
        cfg['decoder_args'].update({'encoding_size': enc_args['target_agent_enc_size']*6})
        wandb.config.update(cfg, allow_val_change=True)
        args.output_dir = os.path.join(args.output_dir, wandb.run.name)


# Make directories
if not os.path.isdir(args.output_dir):
    os.mkdir(args.output_dir)
if not os.path.isdir(os.path.join(args.output_dir, 'checkpoints')):
    os.mkdir(os.path.join(args.output_dir, 'checkpoints'))
if not os.path.isdir(os.path.join(args.output_dir, 'tensorboard_logs')):
    os.mkdir(os.path.join(args.output_dir, 'tensorboard_logs'))

# Initialize tensorboard writer
writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard_logs'))

# Train
trainer = Trainer(cfg, args.data_root, args.data_dir, checkpoint_path=args.checkpoint, writer=writer, wandb_writer=wandb_logger)
trainer.train(num_epochs=int(args.num_epochs), output_dir=args.output_dir)


# Close tensorboard writer
writer.close()
