import sys
import math
import random
import numpy as np

from scipy import ndimage
from skimage.morphology import skeletonize
from typing import Union

from wcode.utils.NDarray_operations import get_largest_k_components


class ScribbleGenerator:
    """
    This one is used to generate scribble label, modified from
    https://github.com/HiLab-git/WSL4MIS/blob/main/code/scribbles_generator.py#L118.
    """

    def __init__(self, ignore_class_id: int, dim: int = 2, seed: int = 319) -> None:
        """
        inputs:
            ignore_class_id: the unlabeled class id in scribble label,
                             can not be 0 and 1 at least, and larger than 0 I think.
        """
        assert ignore_class_id > 1
        self.ignore_class_id = ignore_class_id

        self.dim = dim
        # get the directions
        core = np.ones([3 for _ in range(dim)])
        indexs = np.where(core == 1)
        self.directions = dict()
        for i, position in enumerate(zip(*indexs)):
            self.directions[i] = list(np.array(position) - 1)

        # fix random seed
        np.random.seed(seed)
        random.seed(seed)

        # set the maximum depth of recursive calls
        sys.setrecursionlimit(1000000)

    def generate_scribble(
        self,
        label: np.ndarray,
        iteration: Union[list, tuple],
        cut_branch: bool = True,
    ) -> np.ndarray:
        """
        inputs:
            label: the original label with different classes
            iteration: min and max iterations to do morphology computation for
                       the binary mask of one class. [min_iter, max_iter]
            cut_branch: whether to cut branch for foreground class
        outputs:
            return: generated scribble label for the given label with different classes.
        """
        # foreground classes + background
        number_of_class = np.max(label) + 1
        # initialize the output with unlabeled class value
        output = np.ones_like(label, dtype=np.uint8) * self.ignore_class_id

        for class_id in range(number_of_class):
            if class_id == self.ignore_class_id:
                continue
            scribble = self.get_scribble_for_one_class(
                label, class_id, iteration, cut_branch=cut_branch
            )
            output[scribble == 1] = class_id

        return output

    def get_scribble_for_one_class(
        self,
        label: np.ndarray,
        class_id: int,
        iteration: Union[list, tuple],
        cut_branch: bool = True,
    ) -> np.ndarray:
        """
        inputs:
            label: the original label with different classes
            class_id: value of class in label
            iteration: min and max iterations to do morphology computation for
                       the binary mask of one class. [min_iter, max_iter]
            cut_branch: whether to cut branch for foreground class
        outputs:
            return: generated scribble label for the given label with value 0 and class_id.
        """
        assert isinstance(iteration, (list, tuple))

        binary_mask = label == class_id
        if np.all(binary_mask):
            # If all the pixels in this slice are all belong to the same class,
            # we sample some pixels to make it not to be a line while getting skeleton map.
            binary_mask = self.random_walk(
                binary_mask, path_width=20, scribble_ratio=0.8
            )
        sk_map = self.get_skeleton_map(binary_mask, iteration=iteration)

        # cut the generated skeleton map for foreground class
        if cut_branch and class_id != 0:
            if sk_map.sum() >= 1:
                sk_map = self.cut_branch(sk_map, binary_mask=binary_mask)

        # import matplotlib.pyplot as plt
        # plt.imshow(sk_map.astype(np.uint8)*255, cmap='gray')
        # plt.show()
        # sk_map[sk_map == 0] = self.ignore_class_id
        # sk_map[sk_map != self.ignore_class_id] = class_id
        return sk_map

    def get_skeleton_map(
        self, binary_mask: np.ndarray, iteration: Union[list, tuple]
    ) -> np.ndarray:
        """
        use skeleton map as coarse scribble label
        inputs:
            binary_mask: binary mask of one class, can be 2d or 3d.
                         But recommended to be 2D in the entrie process.
            iteration: min and max iterations to do morphology computation for
                       the binary mask of one class. [min_iter, max_iter]
        outputs:
            return: the skeleton map of the given binary mask
        """
        skeleton_map = np.zeros_like(binary_mask, dtype=np.int32)
        if np.sum(binary_mask) == 0:
            return skeleton_map

        struct = ndimage.generate_binary_structure(self.dim, 2)
        if np.sum(binary_mask) > 900 and len(iteration) == 2:
            iter_num = round(
                iteration[0] + random.random() * (iteration[1] - iteration[0])
            )
            slic = ndimage.binary_erosion(
                binary_mask, structure=struct, iterations=iter_num
            )
        else:
            slic = binary_mask

        return skeletonize(slic, method="lee")

    def cut_branch(
        self, skeleton_map: np.ndarray, binary_mask: np.ndarray, iteration: int = 1
    ) -> np.ndarray:
        """
        Get the binary mask of one class and return the generated scribble label.
        inputs:
            skeleton_map: the generated coarse scribble label
            binary_mask: the binary mask of one class
            iteration: the number to do dilation during postprocess
        outputs:
            output: the refined scribble label
        """
        self.previous_walked_pixels = []
        self.output = np.zeros_like(skeleton_map)
        self.lst_output = np.zeros_like(skeleton_map)
        components = get_largest_k_components(
            skeleton_map, k=2, threshold=15, connectivity=2
        )
        assert isinstance(components, list)

        for c in components:
            start = self.find_start(c)
            self.walk_onebranch(c, start, start)
        self.detect_loop_branch(self.end)

        struct = ndimage.generate_binary_structure(self.dim, 2)
        output = ndimage.binary_dilation(
            self.output, structure=struct, iterations=iteration
        )
        shift = [random.randint(-6, 6) for _ in range(self.dim)]
        rotate = [random.randint(-15, 15) for _ in range(self.dim)]
        if np.sum(binary_mask) > 1000:
            output = ndimage.shift(output.astype(np.uint8), shift=shift)
            if self.dim == 2:
                output = ndimage.rotate(output, rotate[-1], reshape=False)
            elif self.dim == 3:
                axes = [i for i in range(self.dim)]
                for i in axes:
                    output = ndimage.rotate(
                        output, rotate[-1], axes=axes.remove(i), reshape=False
                    )
            else:
                raise RuntimeError(f"Unsupported dimension: {self.dim}")

        return output * binary_mask

    def find_start(self, skeleton_map: np.ndarray) -> tuple:
        """
        find a possible start pixel of the coarse scribble label
        inputs:
            skeleton_map: coarse scribble label need to processed which is a binary mask.
                          Only have spatial channels.
        outputs:
            pixel_index: the index of the selected start point
        """
        # find all the pixels' indexs in the target region
        idxes = np.asarray(np.nonzero(skeleton_map))
        for i in range(idxes.shape[1]):
            pixel_index = tuple([idxes[m, i] for m in range(self.dim)])
            assert skeleton_map[pixel_index] == 1

            # get the possible direction(s) of the selected pixel
            possible_directions = []
            for d_id in self.directions.keys():
                # skip the direction of staying
                if d_id == len(self.directions) // 2:
                    continue
                if self.check_direction(skeleton_map, pixel_index, d_id):
                    possible_directions.append(d_id)

            # if the pixel only have one direction to walk, select it as the start point
            if len(possible_directions) == 1:
                return pixel_index

        # if don't have a pixel which only has one possible direction, just select a pixel at the boundary
        pixel_index = tuple([idxes[m, 0] for m in range(self.dim)])
        return pixel_index

    def check_direction(
        self, skeleton_map: np.ndarray, selected_index: tuple, direct: int
    ) -> bool:
        """
        check whether the pixel at the direction have the same class
        inputs:
            skeleton_map: coarse scribble label need to processed which is a binary mask.
                          Only have spatial channels.
            selected_index: a selected center point
            direct: the direction need to check
        outputs:
            return: whether the pixel at the given direction has the same class
        """
        # the index of the pixel which walk one step forward to the selected direction from the selected pixel.
        walk_forward_index = tuple(
            [selected_index[i] + self.directions[direct][i] for i in range(self.dim)]
        )
        if skeleton_map[walk_forward_index] == 1:
            return True
        elif skeleton_map[walk_forward_index] == 0:
            return False
        else:
            raise ValueError(
                "There cannot be any values other than 0 and 1 in skeleton_map."
            )

    def walk_onebranch(
        self, skeleton_map: np.ndarray, selected_index: tuple, start_index: tuple
    ) -> None:
        """
        walk the whole branch and update self.end
        inputs:
            skeleton_map: coarse scribble label need to processed which is a binary mask.
            selected_index: current coordinates of the random walk
            start_index: start index of this branch
        """
        # find the possible directions
        possible_directions = []
        for d_id in self.directions.keys():
            # skip the direction of staying
            if d_id == len(self.directions) // 2:
                continue
            # skip the direction to the pixel which is already walked on the output.
            if (
                self.output[
                    tuple(
                        [
                            selected_index[i] + self.directions[d_id][i]
                            for i in range(self.dim)
                        ]
                    )
                ]
                == 1
            ):
                continue
            # get the direction to the pixel which is walked on the coarse scribble (skeleton_map),
            # but not on the output (self.output).
            if self.check_direction(skeleton_map, selected_index, d_id):
                possible_directions.append(d_id)

        if len(possible_directions) == 0:
            # already walk to the end pixel of one branch on coarse scribble (skeleton_map).
            self.end = selected_index
        else:
            # select a direction to walk
            walk_direction = random.sample(possible_directions, 1)[0]
            walk_forward_index = tuple(
                [
                    selected_index[i] + self.directions[walk_direction][i]
                    for i in range(self.dim)
                ]
            )
            # If the select pixel dose not make a loop, update self.lst_output
            if len(possible_directions) > 1 and selected_index != start_index:
                self.lst_output = self.output * 1
                self.previous_walked_pixels.append(selected_index)
            self.output[walk_forward_index] = 1
            self.walk_onebranch(skeleton_map, walk_forward_index, start_index)

    def detect_loop_branch(self, end):
        """
        check
        """
        # check all the direction of the ending pixel
        for d_id in self.directions.keys():
            # skip the direction of staying
            if d_id == len(self.directions) // 2:
                continue

            walk_forward_index = tuple(
                [end[i] + self.directions[d_id][i] for i in range(self.dim)]
            )

            if walk_forward_index in self.previous_walked_pixels:
                self.output = self.lst_output * 1

    def random_walk(self, binary_mask, path_width, scribble_ratio, max_iter:int=10000):
        half_size = np.array(binary_mask.shape) // 2
        half_half_size = half_size // 2
        start_point = tuple(
            [
                np.random.randint(
                    half_size[i] - half_half_size[i], half_size[i] + half_half_size[i]
                )
                for i in range(self.dim)
            ]
        )
        walk_path = np.zeros_like(binary_mask)
        if self.dim == 2:
            walk_path[
                start_point[0] : start_point[0] + path_width,
                start_point[1] : start_point[1] + path_width,
            ] = 1
        elif self.dim == 3:
            walk_path[
                start_point[0] : start_point[0] + path_width,
                start_point[1] : start_point[1] + path_width,
                start_point[2] : start_point[2] + path_width,
            ] = 1
        else:
            raise ValueError("Unsupport dismension.")

        num_of_scribble = walk_path.size * scribble_ratio
        iter = 0
        pt = start_point
        while np.sum(walk_path) < num_of_scribble:
            # find possible directions
            possible_directions = []
            for d_id in self.directions.keys():
                # skip the direction of staying
                if d_id == len(self.directions) // 2:
                    continue

                walk_forward_index = tuple(
                    [pt[i] + self.directions[d_id][i] for i in range(self.dim)]
                )

                out_spatial_flag = []
                for i in range(len(walk_forward_index)):
                    if (
                        walk_forward_index[i] < 0
                        or walk_forward_index[i] > binary_mask.shape[i] - path_width
                    ):
                        out_spatial_flag.append(False)
                    else:
                        out_spatial_flag.append(True)

                if np.all(out_spatial_flag):
                    next_value = walk_path[
                        tuple(
                            [
                                pt[i] + self.directions[d_id][i] * path_width
                                for i in range(self.dim)
                            ]
                        )
                    ]
                    if next_value == 0:
                        possible_directions.append(d_id)

            if len(possible_directions) == 0:
                pt = tuple(
                    [
                        np.random.randint(
                            half_size[i] - half_half_size[i],
                            half_size[i] + half_half_size[i],
                        )
                        for i in range(self.dim)
                    ]
                )
                if self.dim == 2:
                    walk_path[
                        pt[0] : pt[0] + path_width,
                        pt[1] : pt[1] + path_width,
                    ] = 1
                elif self.dim == 3:
                    walk_path[
                        pt[0] : pt[0] + path_width,
                        pt[1] : pt[1] + path_width,
                        pt[2] : pt[2] + path_width,
                    ] = 1
                else:
                    raise ValueError("Unsupport dismension.")
            else:
                walk_direction = random.sample(possible_directions, 1)[0]
                pt = tuple(
                    [
                        pt[i] + self.directions[walk_direction][i]
                        for i in range(self.dim)
                    ]
                )
                if self.dim == 2:
                    walk_path[
                        pt[0] : pt[0] + path_width,
                        pt[1] : pt[1] + path_width,
                    ] = 1
                elif self.dim == 3:
                    walk_path[
                        pt[0] : pt[0] + path_width,
                        pt[1] : pt[1] + path_width,
                        pt[2] : pt[2] + path_width,
                    ] = 1
                else:
                    raise ValueError("Unsupport dismension.")

            iter += 1
            if iter > max_iter:
                break

        return walk_path
