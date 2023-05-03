import numpy as np
from preprocessing.util import load_wikiid2nnid
import model.config as config


def keep_only_new_entities(args, folder):
    ent_vecs = np.loadtxt(folder + "ent_vecs/"+args.entity_vectors)
    main_wikiid2nnid = load_wikiid2nnid(txt_file=args.wikiid2nnid)
    additional_wikiids = []
    rows_to_extract = []
    with open(folder + "wikiid2nnid/"+args.wikiid2nnid, "r") as fin, \
        open(folder + "wikiid2nnid/additional_wikiids.txt", "w") as fout:
        for line in fin:
            ent_id, nnid = line.split('\t')
            nnid = int(nnid) - 1  # torch starts from 1 instead of zero
            if ent_id not in main_wikiid2nnid:
                additional_wikiids.append(ent_id)
                rows_to_extract.append(nnid)
                fout.write(ent_id+"\n")
    print("additional entities =", len(additional_wikiids))
    return ent_vecs[rows_to_extract]


def main(args):
    folder = config.base_folder + ("data/entities/extension_entities/" if args.entity_extension else "data/entities/")
    print("folder =", folder)
    #ent_vecs.shape   # (484048, 300)
    if args.entity_extension: ent_vecs = keep_only_new_entities(args, folder)
    else: ent_vecs = np.loadtxt(folder + "ent_vecs/"+args.entity_vectors)
    np.save(folder + "ent_vecs/ent_vecs.npy", ent_vecs)


def _parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity_extension", default=False)
    parser.add_argument("--entity_vectors", default="ent_vecs.txt")
    parser.add_argument("--wikiid2nnid", default="wikiid2nnid.txt")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = _parse_args()
    main(args)
