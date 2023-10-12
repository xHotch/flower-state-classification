"""
CLI for starting the pipeline.
"""

def add_parsers(parser):
    _add_source_parser(parser)


def _add_source_parser(parser):
    parser.add_argument(
        "source",
        nargs="?",
        help="Path to video file or folder with images. If not specified, webcam will be used.",
        default=None,
    )
