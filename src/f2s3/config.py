import json
import logging
from pathlib import Path
from typing import Literal, Optional, Any
from datetime import datetime

import coloredlogs
import numpy as np

from pydantic import BaseModel, ConfigDict, FilePath, DirectoryPath, NewPath, Field, PositiveInt, NonNegativeFloat, \
    model_validator, computed_field, field_validator


class CorrespondenceConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    M: int = 12
    efC: int = 300
    efS: int = 300
    num_threads: int = 16
    space: Literal['l2', 'cosine', 'ip'] = 'l2'
    dimensions: Literal[64] = 64


class F2S3Config(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    # Point cloud files - Can only be None when a tiled_data path is provided
    source: Optional[FilePath] = None
    target: Optional[FilePath] = None

    # Base folder where everything will be saved
    output_folder: DirectoryPath | NewPath = Field(alias='results_dir', default=Path(f"f2s3_results_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}"))

    # Tiling parameters
    start_from_tiled_data: bool = False
    tiled_data: Optional[DirectoryPath] = None
    max_points_per_tile: PositiveInt = 1000000
    min_points_per_tile: PositiveInt = 10000
    overlap_tiles: float = 0.0

    # Supervoxel + Feature Extraction parameters
    batch_size: PositiveInt = 2000
    voxel_grid_size: NonNegativeFloat = 0.0
    max_disp_magnitude: NonNegativeFloat = 0.0
    minimum_points: PositiveInt = 10
    n_normals: PositiveInt = 30 # Numbers of points used to compute normal vectors in the supervoxels

    # Correspondence search parameters
    correspondence_cfg: CorrespondenceConfig = Field(default_factory=CorrespondenceConfig)

    # Post processing parameters
    refine_results: bool = False
    filter_median_magnitude: bool = False
    magnitude_multiplier: float = 30.0
    fill_gaps_c2c: bool = False

    # Output parameters
    save_interim: bool = False
    save_tiles: bool = False
    verbose: bool = False
    num_workers: int = 6

    def feature_radius(self, median_resolution) -> float:
        return float(np.sqrt(3) * (self.minimum_points * median_resolution))

    def supervoxel_radius(self, median_resolution):
        return np.max([ self.feature_radius(median_resolution), self.voxel_grid_size ])

    @field_validator('output_folder', mode='before')
    @classmethod
    def check_output_folder(cls, v: Any):
        v = Path(v)
        v.mkdir(parents=True, exist_ok=True)
        return Path(v).resolve()


    @model_validator(mode='after')
    def check_file_paths(self):
        if self.tiled_data is None:
            self.__dict__['tiled_data'] = self.output_folder / "00_Preprocessing" / "tiles"
            self.__dict__['tiled_data'].mkdir(parents=True, exist_ok=True)

        # Prepare the logger
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)
        coloredlogs.install(level='INFO' if self.verbose else 'VERBOSE', logger=logger)

        for handler in logger.handlers:
            handler.setFormatter(
                logging.Formatter('%(asctime)s [%(levelname)s] %(name)s - %(message)s')
            )

        return self

    def __repr__(self):
        return '\n'.join([f'{field}: {getattr(self, field)}' for field in self.__dataclass_fields__])

    @classmethod
    def load_from_json(cls, file_path: str|Path):
        data = json.load(open(file_path, 'r'))
        return cls(**data)

    def save_to_json(self, file_path: str|Path):
        with open(file_path, 'w') as f:
            data = json.loads(self.model_dump_json())
            json.dump(data, f, indent=4)

    @staticmethod
    def _custom_dir(root_dir: Path, name: str) -> Path:
        folder = root_dir / name
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    @computed_field
    @property
    def interim_dir(self) -> Path:
        return self._custom_dir(self.output_folder, "01_Intermediary")

    @computed_field
    @property
    def result_dir(self) -> Path:
        results_dir = self._custom_dir(self.output_folder, "02_Results")
        if self.refine_results:
            return self._custom_dir(results_dir, "01_refined")
        return results_dir

    @computed_field
    @property
    def supervoxel_dir(self) -> Path|None:
        if self.save_interim:
            return self._custom_dir(self.interim_dir, "02_supervoxels")
        return None

    @computed_field
    @property
    def features_dir(self) -> Path|None:
        if self.save_interim:
            return self._custom_dir(self.interim_dir, "01_features")
        return None

    @computed_field
    @property
    def correspondences_dir(self) -> Path|None:
        if self.save_interim:
            return self._custom_dir(self.interim_dir, "03_correspondences")
        return None

    @computed_field
    @property
    def output_tiles_dir(self) -> Path:
        return self._custom_dir(self.result_dir, "01_tiles")
