import json
import logging
from pathlib import Path
from typing import Literal, Optional

import coloredlogs
import numpy as np

from pydantic import BaseModel, ConfigDict, FilePath, DirectoryPath, NewPath, Field, PositiveInt, NonNegativeFloat, \
    model_validator, computed_field


class CorrespondenceSearchConfig(BaseModel):
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
    source_cloud: Optional[FilePath] = None
    target_cloud: Optional[FilePath] = None

    # Base folder where everything will be saved
    base_dir: DirectoryPath | NewPath = Field(alias='results_dir')

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
    correspondences: CorrespondenceSearchConfig = Field(default_factory=CorrespondenceSearchConfig)

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

    @model_validator(mode='after')
    def check_file_paths(self):
        if not self.start_from_tiled_data:
            if self.source_cloud is None:
                raise FileNotFoundError(f"Source cloud could not be found at {self.source_cloud}!")

            if self.target_cloud is None:
                raise FileNotFoundError(f"Target cloud could not be found at {self.target_cloud}!")
        else:
            if self.tiled_data is None:
                raise NotADirectoryError(f"Tiled data path incorrect: {self.tiled_data}!")

        # Check and set the results path
        if self.base_dir == Path(""):
            self.__dict__['base_dir'] = self.tiled_data.parent if self.start_from_tiled_data else self.source_cloud.parent

        if self.tiled_data is None:
            self.__dict__['tiled_data'] = self.base_dir / "tiled_data"

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
        return self._custom_dir(self.base_dir, "interim")

    @computed_field
    @property
    def result_dir(self) -> Path:
        results_dir = self._custom_dir(self.base_dir, "results")
        if self.refine_results:
            return self._custom_dir(results_dir, "refined")
        return results_dir

    @computed_field
    @property
    def supervoxel_dir(self) -> Path|None:
        if self.save_interim:
            return self._custom_dir(self.interim_dir, "supervoxels")
        return None

    @computed_field
    @property
    def features_dir(self) -> Path|None:
        if self.save_interim:
            return self._custom_dir(self.interim_dir, "features")
        return None

    @computed_field
    @property
    def correspondences_dir(self) -> Path|None:
        if self.save_interim:
            return self._custom_dir(self.interim_dir, "correspondences")
        return None

    @computed_field
    @property
    def output_tiles_dir(self) -> Path:
        return self._custom_dir(self.result_dir, "tiles")
