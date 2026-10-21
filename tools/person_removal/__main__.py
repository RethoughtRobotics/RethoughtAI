"""Remove people from a LeRobot dataset's camera videos. See README.md.

python tools/person_removal masks    --source REPO_OR_DIR --out MASKS
python tools/person_removal remove   --method minimax --source REPO_OR_DIR --masks MASKS --out OUT --models MODELS
python tools/person_removal publish  --folder OUT --repo-id USER/NAME
python tools/person_removal fetch    --method minimax --models MODELS
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # the modules below, run as a directory

CAMERA = 'observation.images.third_eye'


def source_dir(source: str, revision: str | None) -> Path:
    """A local dataset directory as given, else the Hub dataset `source` downloaded (at `revision`)."""
    if Path(source).exists():
        return Path(source)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(source, repo_type='dataset', revision=revision))


def main():
    p = argparse.ArgumentParser(
        prog='person_removal', description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    sub = p.add_subparsers(dest='command', required=True)

    def add_source(q):
        q.add_argument('--source', required=True, help='local LeRobot dataset directory, or a Hub dataset repo id')
        q.add_argument('--revision', help='Hub revision (branch, tag or commit) of --source')
        q.add_argument('--camera', action='append', dest='cameras',
                       help=f'camera feature to process; repeat for several (default {CAMERA})')  # fmt: skip

    q = sub.add_parser('masks', help='1. SAM3 person masks of the cameras')
    add_source(q)
    q.add_argument('--out', type=Path, required=True, help='new directory for the mask videos')
    q.add_argument('--prompt', default='person', help='what SAM3 segments')
    q.add_argument('--threshold', type=float, default=0.3, help='SAM3 instance score threshold')
    q.add_argument('--batch', type=int, default=8, help='frames per SAM3 forward pass')
    q.add_argument('--workers', type=int, default=1, help='parallel processes, spread over the visible GPUs')

    for name, text in (('remove', '2. inpaint the masked person'), ('assemble', '2b. join the parts of sharded runs')):
        q = sub.add_parser(name, help=text)
        add_source(q)
        q.add_argument('--method', required=True, choices=('telea', 'minimax', 'effecterase'))
        q.add_argument('--masks', type=Path, required=True, help='the masks step output')
        q.add_argument('--out', type=Path, required=True, help='new directory for the processed dataset')
        q.add_argument('--parts', type=Path, help='directory of segment parts (with --segment-frames, and assemble)')
        q.add_argument('--steps', type=int, help="denoising steps (default: the method's)")
        q.add_argument('--clip-len', type=int, help='frames per model call (video methods: 4k+1)')
        q.add_argument('--overlap', type=int, help='frames shared by consecutive clips, cross-faded')
        q.add_argument('--dilation', type=int, help='px the mask grows before inpainting')
        q.add_argument('--temporal', type=int, help='frames before and after unioned into each mask')
        q.add_argument('--feather', type=int, help='px over which the output blends into the original')
        if name == 'remove':
            q.add_argument('--models', type=Path, help='directory fetch filled (minimax, effecterase)')
            q.add_argument('--workers', type=int, default=1, help='parallel processes, one model each, over the GPUs')
            q.add_argument('--segment-frames', type=int, default=0,
                           help='cut videos at episode starts into ~N-frame segments, for --shard')  # fmt: skip
            q.add_argument('--shard', default='0/1', help='K/M: this run does every M-th segment from K')
            q.add_argument('--max-frames', type=int, default=0, help='smoke test: only the first N frames per video')

    q = sub.add_parser('publish', help='3. push to the Hub and check every episode decodes')
    q.add_argument('--folder', type=Path, required=True)
    q.add_argument('--repo-id', required=True)
    q.add_argument('--public', action='store_true', help='default: a private repo')
    q.add_argument('--no-check', action='store_true')

    q = sub.add_parser('fetch', help='0. code and weights of a video method')
    q.add_argument('--method', required=True, choices=('minimax', 'effecterase'))
    q.add_argument('--models', type=Path, required=True)

    args = p.parse_args()
    if args.command == 'fetch':
        import fetch

        fetch.fetch(args.method, args.models)
    elif args.command == 'publish':
        import publish

        sha = publish.push(args.folder, args.repo_id, private=not args.public)
        if not args.no_check:
            publish.check(args.repo_id, sha)
    elif args.command == 'masks':
        import masks

        source = source_dir(args.source, args.revision)
        masks.run(source, args.out, args.cameras or [CAMERA], args.workers, args.prompt, args.threshold, args.batch)
    else:
        import remove

        source = source_dir(args.source, args.revision)
        cfg = remove.Settings.for_method(
            args.method, steps=args.steps, clip_len=args.clip_len, overlap=args.overlap, dilation=args.dilation,
            temporal=args.temporal, feather=args.feather, max_frames=getattr(args, 'max_frames', 0) or None,
        )  # fmt: skip
        cameras = args.cameras or [CAMERA]
        if args.command == 'assemble':
            if args.parts is None:
                p.error('assemble needs --parts')
            remove.assemble(source, args.masks, args.out, cameras, cfg, args.parts)
        else:
            if args.method != 'telea' and args.models is None:
                p.error(f'{args.method} needs --models (see: fetch)')
            remove.run(source, args.masks, args.out, cameras, cfg, args.models, args.workers, args.segment_frames,
                       args.shard, args.parts)  # fmt: skip


if __name__ == '__main__':
    main()
