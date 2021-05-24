import argparse

from n3jet.general import FKSModelRun
from n3jet.utils.general_utils import bool_convert

def parse():
    """
    Parse arguments
    """

    parser = argparse.ArgumentParser(description=
                                     """
                                     Training multiple models on the same dataset for error analysis. 
                                     Here we assume that the momenta and njet files already exist and 
                                     will be passed to the script by the user
                                     """
    )

    parser.add_argument(
        '--yaml_file',
        dest='yaml_file',
        help='YAML file with config parameters',
        type=str,
        default = "False"
    )

    parser.add_argument(
        '--mom_file',
        dest='mom_file',
        help='destination of momenta file',
        type=str,
    )

    parser.add_argument(
        '--nj_file',
        dest='nj_file',
        help='NJet file',
        type=str,
    )

    parser.add_argument(
        '--delta_cut',
        dest='delta_cut',
        help='proximity of jets according to JADE algorithm',
        type=float,
        default=0.01,
    )

    parser.add_argument(
        '--delta_near',
        dest='delta_near',
        help='proximity of jets according to JADE algorithm',
        type=float,
    )

    parser.add_argument(
        '--model_base_dir',
        dest='model_base_dir',
        help='model base directory in which folders will be created',
        type=str,
    )

    parser.add_argument(
        '--model_dir',
        dest='model_dir',
        help='model directory which will be created on top of model_base_dir',
        type=str,
    )

    parser.add_argument(
        '--training_reruns',
        dest='training_reruns',
        help='number of training reruns for testing, default: 1',
        type=int,
        default=1,
    )

    parser.add_argument(
        '--all_legs',
        dest='all_legs',
        help='train on data from all legs, not just all jets, default: False',
        type=str,
        default='False',
    )

    parser.add_argument(
        '--all_pairs',
        dest='all_pairs',
        help='train on data from all pairs (except for initial state particles), not just all jets, default: False',
        type=str,
        default='False',
    )

    parser.add_argument(
        '--lr',
        dest='lr',
        help='learning rate',
        type=float,
        default=0.01,
    )

    parser.add_argument(
        '--hp',
        dest='hp',
        help='use float64 precision',
        type=str,
        default="False",
    )

    parser.add_argument(
        '--md',
        dest='md',
        help='train using model_dataset flag',
        type=str,
        default="False",
    )


    args = parser.parse_args()

    return args

if __name__ == "__main__":

    args = parse()
    
    yaml_file = args.yaml_file
    mom_file = args.mom_file
    nj_file = args.nj_file
    delta_cut = args.delta_cut
    delta_near = args.delta_near
    model_base_dir = args.model_base_dir
    model_dir = args.model_dir
    training_reruns = args.training_reruns
    all_legs = bool_convert(args.all_legs)
    all_pairs = bool_convert(args.all_pairs)
    lr = args.lr
    hp = bool_convert(args.hp)
    md = bool_convert(args.md)

    if yaml_file != "False":
        fksmodel = FKSModelRun.from_yaml(yaml_file)
    else:
        fksmodel = FKSModelRun(
            mom_file = mom_file,
            nj_file = nj_file,
            delta_cut = delta_cut,
            delta_near = delta_near,
            model_base_dir = model_base_dir,
            model_dir = model_dir,
            training_reruns = training_reruns,
            all_legs = all_legs,
            all_pairs = all_pairs,
            lr = lr,
            high_precision = hp,
            model_dataset = md
        )

    fksmodel.train()

    

    
