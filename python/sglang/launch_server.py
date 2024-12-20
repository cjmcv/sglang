"""Launch the inference server."""

import os
import sys

from sglang.srt.server import launch_server
from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree

# 主入口： python -m sglang.launch_server 。。。
if __name__ == "__main__":
    
    server_args = prepare_server_args(sys.argv[1:])
    print("hello>>>>>>>>>>>", server_args.show_time_cost)
    try:
        launch_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
