"""Launch the pinned SGLang server with explicitly selected experiment hooks."""
from runtime_opt.expert_scheduling import install_expert_policy
install_expert_policy()
from runtime_opt.tentative_draft import install_tentative_draft
install_tentative_draft()
if __name__ == '__main__':
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.entrypoints.http_server import launch_server
    launch_server(prepare_server_args(__import__('sys').argv[1:]))
