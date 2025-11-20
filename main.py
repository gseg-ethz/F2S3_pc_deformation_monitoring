from f2s3 import F2S3, F2S3Config, CorrespondenceConfig

if __name__ == '__main__':

    cfg = F2S3Config(
        # results_dir=r"/home/jonal/projects/F2S3/data/pchandler_base",
        source=r"/home/jonal/projects/F2S3/data/Mattertal/2019_Ground_aligned_clipped.laz",
        target=r"/home/jonal/projects/F2S3/data/Mattertal/2021_Ground_aligned_clipped.laz",
        start_from_tiled_data=False,
        correspondence_cfg=CorrespondenceConfig()
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