#  pip install pytorch-fid
import subprocess
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="sample images with checkpoint.")
    parser.add_argument("-i", "--dataorg", type=str, default='data/processed/test')
    parser.add_argument("-o", "--dataout", type=str, default='data/output')
    parser.add_argument("--gpu", type=bool, default=False)
    parser.add_argument(
        "--deviceid", type=int, default=1
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    subprocess.run(["python", "-m", "pytorch_fid", args.dataorg,  args.dataout, f'cuda:{args.deviceid}' if args.gpu==True else ""], capture_output=True)
