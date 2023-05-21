import sys
sys.path.append('..')

from src.parser_util import get_parser

options=get_parser().parse_args()
print(options.experiment_root)