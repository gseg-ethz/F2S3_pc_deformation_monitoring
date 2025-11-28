from f2s3 import F2S3, F2S3Config, CorrespondenceConfig

if __name__ == '__main__':

    cfg = F2S3Config(
        source=r"PtCloud1.laz",
        target=r"PtCloud2.laz",
        start_from_tiled_data=False,
    )

    cfg.refine_results = True
    cfg.save_interim = True
    cfg.save_tiles = True
    cfg.filter_median_magnitude = True
    cfg.max_disp_magnitude = 1.0
    cfg.fill_gaps_c2c = True
    cfg.verbose = False

    algorithm = F2S3(cfg)
    algorithm.run()