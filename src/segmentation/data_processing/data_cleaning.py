import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import tifffile
import webdataset as wds
import time



from src.repartition.data_loading import tif_to_real_mask
from src.segmentation.constants import CLASSES_TO_LABELS

def get_annotation_ratio(tif_path: Path) -> float:
    annotation = tifffile.imread(tif_path)  # (C, H, W)
    total_pixels = annotation.shape[1] * annotation.shape[2]
    annotated_pixels = np.sum(np.any(annotation > 0, axis=0))
    return annotated_pixels / total_pixels

def filter_and_compress(dir_path: Path, threshold: float, false_run: bool = False):
    png_files = sorted(dir_path.glob("*.png"))
    print(f"{len(png_files)} images trouvées.")

    valid_pairs = []
    deleted = 0
    missing = 0

    for png_path in png_files:
        tif_path = dir_path / (png_path.stem + ".tif")

        if not tif_path.exists():
            print(f"[NO ANNOTATION]  {png_path.name}")
            missing += 1
            continue

        ratio = get_annotation_ratio(tif_path)

        if ratio < threshold:
            print(f"[DELETE]  {png_path.name} — ratio={ratio:.4f} < seuil={threshold}")
            if not false_run:
                png_path.unlink()
                tif_path.unlink()
            deleted += 1
        else:
            print(f"[KEEP]  {png_path.name} — ratio={ratio:.4f}")
            valid_pairs.append((png_path, tif_path))

    print(f"\nFiltrage : {deleted} paire(s) supprimée(s), {len(valid_pairs)} conservée(s), {missing} annotation(s) manquante(s).")

    if false_run:
        print("[FALSE-RUN] Compression ignorée.")
        return

    if not valid_pairs:
        print("Aucune paire valide à compresser.")
        return
    
    # --- Compression WebDataset ---
    shard_pattern = str(dir_path / "shard-%06d.tar")
    max_shard_size = 1 * 1024 ** 3  # 1 GB

    print(f"\nCompression en shards WebDataset ...")

    with wds.ShardWriter(shard_pattern, maxsize=max_shard_size) as sink:
        for png_path, tif_path in valid_pairs:
            img = np.array(Image.open(png_path).convert("RGB"))
            mask_tif = tifffile.imread(tif_path)
            annotations = tif_to_real_mask(mask_tif)
            mask = np.zeros((img.shape[0], img.shape[1]))
            for cls in CLASSES_TO_LABELS:
                if cls != 'other':
                    mask[annotations[cls] == 255] = CLASSES_TO_LABELS[cls]

            key_name = png_path.stem.replace('.', '_')

            sink.write({
                "__key__": key_name,
                "image.png": img.astype(np.uint8),
                "mask.png": mask.astype(np.uint8),
            })

            png_path.unlink()
            tif_path.unlink()

    print("\nCompression terminée.")

if __name__ == "__main__":
    start = time.time()
    parser = argparse.ArgumentParser(description="Filtre et compresse des paires (image, annotation) en WebDataset.")
    parser.add_argument("-d", "--directory", type=str, help="Répertoire contenant les images .png et annotations .tif.")
    parser.add_argument("-t", "--threshold", type=float, default=0.5, help="Seuil minimal de proportion de pixels annotés (défaut: 0.01 = 1%%).")
    parser.add_argument("-fr", "--false-run", action="store_true", help="Simule sans supprimer ni compresser.")
    args = parser.parse_args()
    main_dir = Path(args.directory)
    for path in main_dir.iterdir():
        if path.is_dir():
            filter_and_compress(path, args.threshold, false_run=args.false_run)
    end = time.time()
    print("Done.")
    print(f"Execution time : {(end - start)/60} min")
