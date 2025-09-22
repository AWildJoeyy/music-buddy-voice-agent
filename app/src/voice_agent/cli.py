# src/cli.py
import argparse
import sys
import sounddevice as sd

from voice_agent.asr.app import main as asr_main
from voice_agent.asr.backend_whisper import WhisperBackend


def _forward_to_asr(args: argparse.Namespace):
    """Forward parsed args to asr.app:main so the parsing logic lives in one place."""
    argv = ["va-asr"]
    for k, v in vars(args).items():
        if k == "cmd":
            continue
        flag = f"--{k.replace('_','-')}"
        if isinstance(v, bool):
            if v:  # include only True flags
                argv.append(flag)
        elif v is not None:
            argv.extend([flag, str(v)])
    sys.argv = argv
    return asr_main()


def main():
    p = argparse.ArgumentParser(prog="va", description="Voice Agent CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    # ---- ASR (live mic) ----
    a = sub.add_parser("asr", help="Run always-on ASR (faster-whisper + energy VAD)")
    a.add_argument("--model", default="small.en")
    a.add_argument("--device-type", default="auto", choices=["auto", "cpu", "cuda"])
    a.add_argument("--compute-type", default="auto")
    a.add_argument("--lang", default="en")
    a.add_argument("--beam-size", type=int, default=5)
    a.add_argument("--gain-db", type=float, default=0.0)

    a.add_argument("--frame-ms", type=int, default=20, choices=[10, 20, 30])
    a.add_argument("--silence-tail-ms", type=int, default=400)
    a.add_argument("--rms-floor", type=float, default=60.0)
    a.add_argument("--thr-mult", type=float, default=1.6)
    a.add_argument("--warmup-ms", type=int, default=400)
    a.add_argument("--pre-roll-ms", type=int, default=240)
    a.add_argument("--min-utter-ms", type=int, default=300)

    a.add_argument("--device", type=int, default=None, help="Input device index")
    a.add_argument("--jsonl", default=None)
    a.add_argument("--no-clip", action="store_true")

    # ---- ASR (file) ----
    f = sub.add_parser("asr-file", help="Transcribe an audio file with faster-whisper")
    f.add_argument("--input", required=True, help="Path to audio file (wav/mp3/m4a...)")
    f.add_argument("--model", default="small.en")
    f.add_argument("--device-type", default="auto", choices=["auto", "cpu", "cuda"])
    f.add_argument("--compute-type", default="auto")
    f.add_argument("--lang", default="en")
    f.add_argument("--beam-size", type=int, default=5)

    # ---- List audio input devices ----
    sub.add_parser("devices", help="List audio input devices")

    args = p.parse_args()

    if args.cmd == "devices":
        default_in = None
        try:
            default_pair = sd.default.device  # (input_idx, output_idx)
            default_in = default_pair[0] if isinstance(default_pair, (list, tuple)) else default_pair
        except Exception:
            pass

        for idx, dev in enumerate(sd.query_devices()):
            ins = dev.get("max_input_channels", 0)
            if ins > 0:
                star = " *" if default_in == idx else ""
                print(f"{idx:>3}  {dev['name']}  (inputs: {ins}){star}")
        return

    if args.cmd == "asr":
        return _forward_to_asr(args)

    if args.cmd == "asr-file":
        b = WhisperBackend(
            model_name=args.model,
            device=args.device_type,
            compute_type=args.compute_type,
            language=args.lang,
            beam_size=args.beam_size,
            vad_filter=True,
        )
        print(b.transcribe_file(args.input))
        return


if __name__ == "__main__":
    main()
