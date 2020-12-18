import os
from ele_common.units import Generator
from gooey import Gooey, GooeyParser


@Gooey(
    advanced=True,
    auto_start=True,
    program_name='electronic data generator',
)
def main():

    parser = GooeyParser()

    parser.add_argument(
        "generate_directory",
        help="Path to the directory you want to generate data",
        widget="DirChooser",
        # metavar="METAVAR"
    )

    parser.add_argument(
        "layer_number",
        help="The number of layer use",
        type=int,
        default=3,
    )

    args = parser.parse_args()


if __name__ == '__main__':
    main()