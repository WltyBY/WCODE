# Copyright [2019] [Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany]
import numpy as np


def get_patch_size(final_patch_size, rot_x, rot_y, rot_z, scale_range):
    if isinstance(rot_x, (tuple, list)):
        rot_x = max(np.abs(rot_x))
    if isinstance(rot_y, (tuple, list)):
        rot_y = max(np.abs(rot_y))
    if isinstance(rot_z, (tuple, list)):
        rot_z = max(np.abs(rot_z))
    rot_x = min(90 / 360 * 2.0 * np.pi, rot_x)
    rot_y = min(90 / 360 * 2.0 * np.pi, rot_y)
    rot_z = min(90 / 360 * 2.0 * np.pi, rot_z)
    coords = np.array(final_patch_size)
    final_shape = np.copy(coords)
    if len(coords) == 3:
        final_shape = np.max(
            np.vstack((np.abs(rotate_coords_3d(coords, rot_x, 0, 0)), final_shape)), 0
        )
        final_shape = np.max(
            np.vstack((np.abs(rotate_coords_3d(coords, 0, rot_y, 0)), final_shape)), 0
        )
        final_shape = np.max(
            np.vstack((np.abs(rotate_coords_3d(coords, 0, 0, rot_z)), final_shape)), 0
        )
    elif len(coords) == 2:
        final_shape = np.max(
            np.vstack((np.abs(rotate_coords_2d(coords, rot_x)), final_shape)), 0
        )
    final_shape /= min(scale_range)
    return final_shape.astype(int)


def rotate_coords_3d(coords, angle_x, angle_y, angle_z):
    rot_matrix = np.identity(len(coords))
    rot_matrix = create_matrix_rotation_x_3d(angle_x, rot_matrix)
    rot_matrix = create_matrix_rotation_y_3d(angle_y, rot_matrix)
    rot_matrix = create_matrix_rotation_z_3d(angle_z, rot_matrix)
    coords = (
        np.dot(coords.reshape(len(coords), -1).transpose(), rot_matrix)
        .transpose()
        .reshape(coords.shape)
    )
    return coords


def rotate_coords_2d(coords, angle):
    rot_matrix = create_matrix_rotation_2d(angle)
    coords = (
        np.dot(coords.reshape(len(coords), -1).transpose(), rot_matrix)
        .transpose()
        .reshape(coords.shape)
    )
    return coords


def create_matrix_rotation_x_3d(angle, matrix=None):
    rotation_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)],
        ]
    )
    if matrix is None:
        return rotation_x

    return np.dot(matrix, rotation_x)


def create_matrix_rotation_y_3d(angle, matrix=None):
    rotation_y = np.array(
        [
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ]
    )
    if matrix is None:
        return rotation_y

    return np.dot(matrix, rotation_y)


def create_matrix_rotation_z_3d(angle, matrix=None):
    rotation_z = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    if matrix is None:
        return rotation_z

    return np.dot(matrix, rotation_z)


def create_matrix_rotation_2d(angle, matrix=None):
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    if matrix is None:
        return rotation

    return np.dot(matrix, rotation)
