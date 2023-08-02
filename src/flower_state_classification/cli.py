def add_parsers(parser):
    _add_source_parser(parser)


def _add_source_parser(parser):
    parser.add_argument(
        "source",
        nargs="?",
        help="Path to video file or folder with images. If not specified, webcam will be used.",
        default=None,
    )

    parser.add_argument(
        "pipeline_mode",
        nargs="?",
        help="Specify wether the pipeline is run in scheduled or continuous mode. If not specified, the default values from settings.py will be used.",
        default=None,
    )
