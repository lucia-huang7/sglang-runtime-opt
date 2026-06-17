"""Generate the version-specific opt-in verification snapshot patch."""
import argparse,difflib,hashlib,json
from pathlib import Path


def patch(text):
    if hashlib.sha256(text.encode()).hexdigest() != 'c7930a0361a479c3405c4e7c28a84f251cb92b108b851903f82fe4552648cb37':
        raise ValueError('Verification source differs from the pinned upstream revision')
    marker='        accept_index_cpu = accept_index.tolist()\n        predict_cpu = predict.tolist()'
    if text.count(marker)!=1:raise ValueError('Unexpected verification CPU-consumer source')
    replacement='''        acceptance_cap = __import__("os").environ.get("SGLANG_VERIFY_ACCEPTANCE_CAP")
        if acceptance_cap is not None:
            cap = int(acceptance_cap)
            if not 0 <= cap <= self.spec_steps:
                raise ValueError("Invalid verification acceptance cap")
            positions = torch.arange(accept_index.shape[1], device=accept_index.device)
            accept_index = torch.where(positions[None, :] > cap, -1, accept_index)
            accept_length = accept_length.clamp_max(cap)
        packed_accept_length_cpu = None
        if __import__("os").environ.get("SGLANG_PACKED_VERIFY_D2H") == "1":
            from runtime_opt.verification_transfer import VerificationSnapshot
            from runtime_opt.tentative_draft import current_worker, eligible, start_tentative
            worker = current_worker()
            can_tentatively_extend = eligible(worker, batch, self)
            transfer_stream = None
            if can_tentatively_extend:
                if not hasattr(worker, "_runtime_verify_stream"):
                    worker._runtime_verify_stream = torch.cuda.Stream(device=predict.device)
                transfer_stream = worker._runtime_verify_stream
            transfer = VerificationSnapshot(accept_index, predict, accept_length, stream=transfer_stream)
            if can_tentatively_extend:
                start_tentative(batch, self, target_predict)
            accept_index_cpu, predict_cpu, packed_accept_length_cpu = transfer.result()
        else:
            accept_index_cpu = accept_index.tolist()
            predict_cpu = predict.tolist()'''
    text=text.replace(marker,replacement)
    marker='                accept_length_cpu=accept_length.tolist(),'
    if text.count(marker)!=1:raise ValueError('Unexpected accepted-length source')
    text=text.replace(marker,'''                accept_length_cpu=(packed_accept_length_cpu if packed_accept_length_cpu is not None
                                   else accept_length.tolist()),''')
    return text

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--apply',action='store_true');p.add_argument('--diff',required=True);a=p.parse_args()
    path=Path(a.source);before=path.read_text();after=patch(before)
    compile(after,str(path),'exec')
    Path(a.diff).write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),
        fromfile='a/python/sglang/srt/speculative/eagle_utils.py',tofile='b/python/sglang/srt/speculative/eagle_utils.py')))
    print(json.dumps({'original_sha256':hashlib.sha256(before.encode()).hexdigest(),'patched_sha256':hashlib.sha256(after.encode()).hexdigest()}))
    if a.apply:path.write_text(after)
