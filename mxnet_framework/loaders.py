from os.path import dirname, join
from image_sample import SampleWithCache
import multiprocessing
from functools import partial

def _parse_line(line, cfg, imgs_dir):
    parts = line.split(';')
    local_path = parts[0]

    img_path = join(imgs_dir, local_path)
    image_sample = SampleWithCache(img_path, parts[1:], cfg)

    return image_sample

def load_samples(db_params, cfg, csv_workers):
    samples = []
    for db_param in db_params:
        assert(db_param['TYPE'] in ('CSV_FILE'))
        if db_param['TYPE'] == 'CSV_FILE':
            csv_path = join(db_param['DB_PATH'], db_param['CSV_FILE'])
            _samples = load_images_from_csv(csv_path, cfg, csv_workers)
            samples.extend(_samples)
    return samples

def load_images_from_csv(csv_path, cfg, csv_workers):
    imgs_dir = dirname(csv_path)
    with open(csv_path) as fp:
        content = fp.read().splitlines()

    if csv_workers == 0:
        samples = [_parse_line(line, cfg, imgs_dir) for line in content]
    else:
        pool = multiprocessing.Pool(processes=csv_workers)
        samples = pool.map(partial(_parse_line, cfg=cfg, imgs_dir=imgs_dir), content)
        pool.close()
        pool.join()
        assert(len(samples) == len(content))

    return samples