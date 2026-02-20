import argparse
import json
import sys

from dotenv import load_dotenv

# Load .env before importing app.graph — the OpenAI client reads env vars at init
load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agentic RAG Resume — ask questions about Alain Feigneux's background."
    )
    parser.add_argument("question", help="The question to answer")
    parser.add_argument(
        "--mode",
        choices=["qa", "executive", "incident"],
        default="qa",
        help="Response mode: qa (default), executive, or incident",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of chunks to retrieve from MongoDB (default: 5)",
    )
    parser.add_argument(
        "--voice",
        action="store_true",
        default=False,
        help="Speak the short_answer aloud via ElevenLabs TTS",
    )
    args = parser.parse_args()

    try:
        from app.graph import run_graph

        result = run_graph(
            question=args.question, mode=args.mode, voice=args.voice, k=args.k
        )
        print(json.dumps(result, indent=2))
        sys.exit(0)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        print(f"FATAL: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
