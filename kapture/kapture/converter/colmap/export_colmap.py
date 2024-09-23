# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
This script exports data from kapture to COLMAP database and/or reconstruction files.
"""

import logging
import os
import os.path as path
from typing import Optional
import warnings

# kapture
import kapture
from kapture.core.Trajectories import rigs_remove_inplace
import kapture.algo.merge_keep_ids
import kapture.io.csv as csv
import kapture.io.structure
import kapture.utils.paths
from kapture.utils.Collections import try_get_only_key_from_collection

# local
from .database import COLMAPDatabase
from .database_extra import is_colmap_db_empty, kapture_to_colmap, get_colmap_image_ids_from_db,\
    get_colmap_camera_ids_from_db
from .export_colmap_rigs import export_colmap_rig
from .export_colmap_reconstruction import export_to_colmap_txt

logger = logging.getLogger('colmap')


def export_colmap(kapture_dir_path: str,
                  colmap_database_filepath: str,
                  colmap_reconstruction_dir_path: Optional[str],
                  keypoints_type: Optional[str] = None,
                  descriptors_type: Optional[str] = None,
                  colmap_rig_filepath: str = None,
                  force_overwrite_existing: bool = False,
                  pairsfile_path: Optional[str] = None) -> None:
    """
    Exports kapture data to colmap database and or reconstruction text files.

    :param kapture_dir_path: kapture top directory
    :param colmap_database_filepath: path to colmap database file
    :param colmap_reconstruction_dir_path: path to colmap reconstruction directory
    :param keypoints_type: type of keypoints, name of the keypoints subfolder
    :param descriptors_type: type of descriptors to export, name of the descriptors subfolder
    :param colmap_rig_filepath: path to colmap rig file
    :param force_overwrite_existing: Silently overwrite colmap files if already exists.
    :param pairsfile_path: Filter matches to load
    """

    os.makedirs(path.dirname(colmap_database_filepath), exist_ok=True)
    if colmap_reconstruction_dir_path:
        os.makedirs(colmap_reconstruction_dir_path, exist_ok=True)

    assert colmap_database_filepath
    kapture.utils.paths.safe_remove_file(colmap_database_filepath, force_overwrite_existing, 'colmap database')

    logger.info('creating colmap database in {}'.format(colmap_database_filepath))
    db = COLMAPDatabase.connect(colmap_database_filepath)
    if not is_colmap_db_empty(db):
        raise ValueError('the existing colmap database is not empty : {}'.format(colmap_database_filepath))

    logger.info('loading kapture files...')
    with csv.get_all_tar_handlers(kapture_dir_path) as tar_handlers:
        kapture_data = csv.kapture_from_dir(kapture_dir_path, pairsfile_path, tar_handlers=tar_handlers)
        assert isinstance(kapture_data, kapture.Kapture)

        if keypoints_type is None:
            keypoints_type = try_get_only_key_from_collection(kapture_data.keypoints)
        # if it's still None try again from matches
        if keypoints_type is None:
            keypoints_type = try_get_only_key_from_collection(kapture_data.matches)
        if descriptors_type is None:
            descriptors_type = try_get_only_key_from_collection(kapture_data.descriptors)

        # COLMAP does not fully support rigs.
        if kapture_data.rigs is not None:
            # make sure, rigs are not used in trajectories.
            logger.info('remove rigs notation.')
            rigs_remove_inplace(kapture_data.trajectories, kapture_data.rigs)

        # write colmap database
        kapture_to_colmap(kapture_data, kapture_dir_path, tar_handlers, db, keypoints_type, descriptors_type)

        if colmap_reconstruction_dir_path:
            # create text files
            colmap_camera_ids = get_colmap_camera_ids_from_db(db, kapture_data.records_camera)
            colmap_image_ids = get_colmap_image_ids_from_db(db)
            export_to_colmap_txt(colmap_reconstruction_dir_path,
                                 kapture_data,
                                 kapture_dir_path,
                                 tar_handlers,
                                 colmap_camera_ids,
                                 colmap_image_ids,
                                 keypoints_type)
            if colmap_rig_filepath:
                try:
                    if kapture_data.rigs is None:
                        raise ValueError('No rig to export.')
                    export_colmap_rig(colmap_rig_filepath,
                                      kapture_data.rigs,
                                      kapture_data.records_camera,
                                      colmap_camera_ids)
                except Exception as e:
                    warnings.warn(e)
                    logger.warning(e)

    db.close()
