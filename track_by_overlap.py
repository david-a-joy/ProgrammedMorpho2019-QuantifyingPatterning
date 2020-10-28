#!/usr/bin/env python3

""" Track cells in a partially labeled image by overlap

To track all frames of a time series in a single folder:

.. code-block:: bash

    $ python3 track_by_overlap.py [/path/to/folder]

To track all frames of a time series and write the results to a folder:

.. code-block:: bash

    $ python3 track_by_overlap.py -o [/path/to/results] [/path/to/folder]

To change the threshold used to segment the images

.. code-block:: bash

    $ python3 track_by_overlap.py -t [threshold] [/path/to/folder]

"""

# Imports

# Standard lib
import copy
import shutil
import pathlib
import argparse
from typing import Tuple, List, Dict, Optional

# 3rd party
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

from scipy import ndimage as ndi

from skimage.filters import gaussian
from skimage.feature import peak_local_max
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed
from skimage.measure import find_contours
from skimage.io import imread

from sklearn.mixture import GaussianMixture

# Smoothing Constants
FOREGROUND_SMOOTHING: Tuple[float] = (3, 3)  # Gaussian smoothing for the foreground image
BACKGROUND_SMOOTHING: Tuple[float] = (30, 30)  # Gaussian smoothing to estimate the image background

# Segmentation Constants
# THRESHOLD_ABS: Optional[float] = 0.05  # Threshold between background and foreground after smoothing
THRESHOLD_ABS: Optional[float] = None  # Set to None to estimate a good threshold
EXCLUDE_BORDER: int = 25  # px - remove objects too close to the border
MIN_OBJECT_SIZE: int = 500  # px - minimum size for objects
FOOTPRINT_SIZE: int = 25  # px - Side length of the square region to search for peaks

# Linking Constants
MIN_OVERLAP: int = 10  # px - minimum area overlap between segmentations


# Helper Functions


def match_labels_by_overlap(curr_labels: np.ndarray,
                            prev_labels: np.ndarray,
                            min_overlap: int = MIN_OVERLAP) -> np.ndarray:
    """ Match labels with overlap

    :param ndarray curr_labels:
        The current labels in this frame
    :param ndarray prev_labels:
        The labels in the previous frame
    :param int min_overlap:
        Minimum acceptable overlap between regions in pixels
    :returns:
        A new array with indexing consistent between curr_labels and prev_labels
    """
    # Number all the regions, with unmatched regions getting a higher number
    next_label: int = np.max(prev_labels) + 1

    # Make a copy of the old and new arrays
    prev_labels = prev_labels.copy()
    final_labels = np.zeros_like(curr_labels)

    # First sort all the labels by size, checking larger labels first
    label_order = ((i, np.sum(curr_labels == i))
                   for i in np.unique(curr_labels)
                   if i > 0)  # Ignore the background label

    for label_index, label_size in sorted(label_order, reverse=True, key=lambda x: x[1]):
        # Drop any labels that are too small
        if label_size < min_overlap:
            continue

        label_mask = curr_labels == label_index

        # Find all the labels that intersect this label
        prev = prev_labels[label_mask]

        # Any labels with no overlap with any other label must be new cells
        if np.sum(prev > 0) < min_overlap:
            final_labels[label_mask] = next_label
            next_label += 1
            continue

        # Find the most frequent index in this overlapping region
        unique_indices, counts = np.unique(prev, return_counts=True)
        best_mask = np.logical_and(unique_indices > 0, counts >= min_overlap)

        # If we failed to match any indices, this is a new cell
        if not np.any(best_mask):
            final_labels[label_mask] = next_label
            next_label += 1
            continue

        # Find the best non-background index that has enough overlap
        counts = counts[best_mask]
        unique_indices = unique_indices[best_mask]
        best = unique_indices[np.argsort(counts)[-1]]

        # This region should be labeled with the best single old label
        final_labels[label_mask] = best

        # Zero the label in the previous mask, so we don't link twice
        prev_labels[prev_labels == best] = 0

    return final_labels


# Classes


class TrackData(object):
    """ Store data for a single track over time

    :param int track_idx:
        The index for this track
    """

    def __init__(self, track_idx: int):
        self.track_idx = track_idx

        # Which frames is the track defined for
        self.frames: List[int] = []

        # Where are the centers of mass for each frame
        self.centers: List[Tuple[float]] = []

        # Where are the perimeters for each frame
        self.perimeters: List[np.ndarray] = []

    def __len__(self) -> int:
        return len(self.frames)

    def __repr__(self) -> str:
        return f'Track({self.track_idx}, {len(self)} frames)'

    def plot_track(self,
                   frame_idx: int,
                   ax: plt.Axes,
                   max_track_idx: int,
                   arrow_scale: float = 4.0):
        """ Plot the track on this frame

        :param int frame_idx:
            The frame index to plot the track for
        :param Axes ax:
            The matplotlib axis to plot this track on
        :param int max_track_idx:
            Maximum track index to plot (sets this track's color)
        :param float arrow_scale:
            Scale factor to exaggerate the velocity arrows by
        """

        # Only plot frames where this track is found
        if frame_idx not in self.frames:
            return

        # Find the index in this track that corresponds to the global frame number
        this_frame = self.frames.index(frame_idx)

        # Work out the color for this track using the track index and maximum track number
        cmap = copy.copy(get_cmap('Set1'))
        cmap.set_under('black')
        rgba = cmap((self.track_idx - 1)/(max_track_idx))

        # Load the data for this frame
        cx, cy = self.centers[this_frame]
        perimeter = self.perimeters[this_frame]

        # Try to calculate velocity, if we can
        dx, dy = self.calc_delta(frame_idx)

        # Plot center of mass and perimeter
        ax.plot([cy], [cx], '.', color=rgba, linestyle='')
        ax.plot(perimeter[:, 1], perimeter[:, 0], color=rgba, linestyle='-', linewidth=2)

        # Plot velocity arrows, if available
        if dx is not None and dy is not None:
            ax.arrow(cy, cx, dy*arrow_scale, dx*arrow_scale,
                     color=rgba, linestyle='-', linewidth=2, width=0.01)

    def append(self, frame_idx: int, frame: np.ndarray):
        """ Append a segmented region from a frame """

        # Pull the track index out of this frame segmentation
        mask = frame == self.track_idx

        # Not found, so don't add the track
        if not np.any(mask):
            return

        # Pull out the center of mass
        cx, cy = ndi.center_of_mass(mask)

        # Pull out the contour around the region
        perimeter = find_contours(mask, 0.5)[0]

        self.frames.append(frame_idx)
        self.centers.append((cx, cy))
        self.perimeters.append(perimeter)

    def calc_delta(self, frame_idx: int) -> Tuple[Optional[float]]:
        """ Calculate velocity

        :param int frame_idx:
            The frame to calculate dx, dy pairs for
        :returns:
            The delta x, delta y tuple for this frame (or None, None if not possible)
        """

        # Work out which other frames we have tracks for
        if frame_idx - 1 in self.frames:
            prev_frame = self.frames.index(frame_idx - 1)
        else:
            prev_frame = None
        if frame_idx + 1 in self.frames:
            next_frame = self.frames.index(frame_idx + 1)
        else:
            next_frame = None

        # No information, return None
        if prev_frame is None and next_frame is None:
            return None, None

        # Forward differences
        if prev_frame is None:
            cx, cy = self.centers[self.frames.index(frame_idx)]
            nx, ny = self.centers[next_frame]
            return (nx - cx), (ny - cy)

        # Backwards differences
        if next_frame is None:
            cx, cy = self.centers[self.frames.index(frame_idx)]
            px, py = self.centers[prev_frame]
            return (cx - px), (cy - py)

        # Everything, so central differences
        nx, ny = self.centers[next_frame]
        px, py = self.centers[prev_frame]
        return (nx - px)/2.0, (ny - py)/2.0


class OverlapTracker(object):
    """ Track objects in an mCherry time lapse by overlaping contours

    :param Path image_dir:
        Directory containing the .tif image frames, in sorter by alphnumeric sort
    :param float threshold_abs:
        If not None, the threshold to segment all images with
        If None, estimate this threshold from the first frame
    :param int min_object_size:
        Minimum size (in pixels) of segmented objects to keep
    :param int exclude_border:
        Minimum distance from the image border (in pixels) to allow objects
    """

    def __init__(self,
                 image_dir: pathlib.Path,
                 threshold_abs: Optional[float] = THRESHOLD_ABS,
                 min_object_size: int = MIN_OBJECT_SIZE,
                 exclude_border: int = EXCLUDE_BORDER,
                 footprint_size: int = FOOTPRINT_SIZE):

        # Path to a directory containing tifs ordered by frame number
        self.image_dir: pathlib.Path = image_dir

        # Configuration parameters
        self.threshold_abs: Optional[float] = threshold_abs
        self.min_object_size: int = min_object_size
        self.exclude_border: int = exclude_border
        self.footprint_size: int = footprint_size

        # Unprocessed image frames
        self.frames: List[np.ndarray] = []

        # Background corrected image frames
        self.smooth_frames: List[np.ndarray] = []

        # Segmented frames using a threshold and a watershed operation
        self.segmented_frames: List[np.ndarray] = []

        # Linked frames using area intersection
        self.linked_frames: List[np.ndarray] = []

        # Track stats extracted from the labels
        self.track_stats: List[TrackData] = []

    def load_frames(self):
        """ Load individual frame images """
        # Load all the individual frames into a list of arrays, ignoring non-image files
        # All frames must be sortable by name, e.g. frame0001.tif, frame0002.tif, etc
        for frame_file in sorted(self.image_dir.iterdir()):
            if frame_file.suffix not in ('.png', '.tif'):
                continue
            img = imread(frame_file).astype(np.float)
            if img.ndim == 3:
                img = np.mean(img, axis=2)
            if img.ndim != 2:
                raise ValueError(f'Expected 2D or 3D color image, got {img.shape}')
            img_min = np.min(img)
            img_max = np.max(img)
            img = (img - img_min) / (img_max - img_min)
            self.frames.append(img)

        if len(self.frames) < 1:
            raise OSError(f'No valid frames found under {self.image_dir}')

    def background_correct_frames(self):
        """ Perform background subtraction to get a smooth frame """

        for frame in self.frames:
            # Background subtraction
            # Increase this constant to smooth foreground noise, decrease to sharpen boundaries
            fg_frame = gaussian(frame, FOREGROUND_SMOOTHING)
            # Increase this constant to keep larger background features, decrease to remove features
            bg_frame = gaussian(frame, BACKGROUND_SMOOTHING)

            # Subtract then cap the scale of the activation
            smooth_frame = fg_frame - bg_frame
            smooth_frame[smooth_frame < 0] = 0
            smooth_frame[smooth_frame > 1] = 1

            self.smooth_frames.append(smooth_frame)

    def segment_frames(self):
        """ Segment smoothed frames using a threshold followed by a watershed """

        # Estimate the threshold from the image, if not user defined
        if self.threshold_abs is None:
            self.threshold_abs = self.estimate_threshold()

        footprint = np.ones((self.footprint_size, self.footprint_size))

        # Threshold and segment
        for smooth_frame in self.smooth_frames:
            mask_frame = smooth_frame > self.threshold_abs

            # No detection, return an empty mask
            if not np.any(mask_frame):
                self.segmented_frames.append(mask_frame.astype(np.int))
                continue

            # Remove particles smaller than the min size
            mask_frame = remove_small_objects(mask_frame, min_size=self.min_object_size)

            # Remove detections too close to the border
            mask_frame[:self.exclude_border, :] = 0
            mask_frame[:, :self.exclude_border] = 0
            mask_frame[-self.exclude_border:, :] = 0
            mask_frame[:, -self.exclude_border:] = 0

            # Maybe we zeroed the mask here, so again check for no detection
            if not np.any(mask_frame):
                self.segmented_frames.append(mask_frame.astype(np.int))
                continue

            # Split the objects using the watershed algorithm
            dist_frame = ndi.distance_transform_edt(mask_frame)
            local_maxi = peak_local_max(dist_frame, indices=False,
                                        footprint=footprint,
                                        labels=mask_frame)
            markers = ndi.label(local_maxi)[0]
            labels = watershed(-dist_frame, markers, mask=mask_frame)

            # One more cleaning pass to remove noise
            labels = remove_small_objects(labels, min_size=self.min_object_size)

            self.segmented_frames.append(labels)

    def link_frames(self):
        """ Connect frame segmentations over time """

        # Renumber all the regions using the numbers in the previous frame
        # This links all the regions over time, making track extraction simple
        prev_frame: Optional[np.ndarray] = None

        for curr_frame in self.segmented_frames:
            # For the first frame, just keep whatever values we have
            if prev_frame is None:
                self.linked_frames.append(curr_frame)
                prev_frame = curr_frame
                continue
            # Copy any overlapping regions from the last image to the next
            final_frame = match_labels_by_overlap(curr_frame, prev_frame)

            # Write the links to the frame stack
            self.linked_frames.append(final_frame)
            prev_frame = final_frame

    def calculate_stats(self):
        """ Using the linked frames, calculate center of mass, velocity, and perimeter """

        # All the linked use the same indicies, so we can extract tracks by just counting
        tracks: Dict[int, TrackData] = {}

        for frame_idx, frame in enumerate(self.linked_frames):

            # Loop over any indicies in this frame
            for track_idx in np.unique(frame):
                if track_idx == 0:
                    continue
                # Get or copy the track data
                track = tracks.get(track_idx, TrackData(track_idx))
                track.append(frame_idx, frame)
                tracks[track_idx] = track

        self.track_stats = list(tracks.values())

    def calculate_single_stats(self):
        """ Calculate stats without linking frames """

        # All the linked use the same indicies, so we can extract tracks by just counting
        tracks: Dict[int, TrackData] = {}

        for frame_idx, frame in enumerate(self.segmented_frames):

            # Loop over any indicies in this frame
            for track_idx in np.unique(frame):
                if track_idx == 0:
                    continue

                # Always assign the track a new id
                track_idx = len(tracks)

                # Every frame is a new track
                track = TrackData(track_idx)
                track.append(frame_idx, frame)
                tracks[track_idx] = track

        self.track_stats = list(tracks.values())

    def plot_individual_frames(self, outdir: Optional[pathlib.Path] = None):
        """ Plot the tracks on individual frames

        :param Path outdir:
            If not None, save the frames to an output directory
            If None, display each frame in matplotlib
        """
        # Create a fresh directory to write the frames to
        if outdir is not None:
            if outdir.is_dir():
                shutil.rmtree(str(outdir))
            outdir.mkdir(parents=True)

        # Work out the highest track number among tracks
        max_track_idx = np.max([t.track_idx for t in self.track_stats])

        for frame_idx, frame in enumerate(self.frames):

            # Plot the original image
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            ax.imshow(frame, cmap='gray')

            # Try to plot as many tracks as possible on the image
            for track in self.track_stats:
                track.plot_track(frame_idx, ax, max_track_idx)

            # Remove borders and other decorations from the frame
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim([0, frame.shape[1]])
            ax.set_ylim([frame.shape[0], 0])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # Make the plot as large and compact as possible
            plt.tight_layout()

            if outdir is None:
                # If no output directory was passed, save the frame
                plt.show()
            else:
                # If the output directory was passed, save this frame to it
                outfile = '{:05d}.svg'.format(frame_idx)
                fig.savefig(str(outdir / outfile),
                            transparent=True)
                plt.close()

    # Helper methods

    def estimate_threshold(self, frame_index: int = 0) -> float:
        """ Select a threshold for segmentation

        :param int frame_index:
            Which frame to use to estimate the threshold from
        :returns:
            The 2-mode mixture model segmentation threshold
        """
        # Work out the segmentation boundary
        gmm = GaussianMixture(n_components=2)
        gmm.fit(self.smooth_frames[frame_index].reshape(-1, 1))

        # Find the larger of the two means and use that for the threshold
        m1, m2 = gmm.means_.ravel()
        threshold_abs = np.max([m1, m2])
        print(f'Selecting threshold: {threshold_abs}')
        return threshold_abs


# Command line interface

if __name__ == '__main__':
    # Parse the supplied command line arguments
    parser = argparse.ArgumentParser('Track objects by overlap')
    parser.add_argument('-o', '--outdir', type=pathlib.Path,
                        help='Directory to write tracked frames to (default is to plot them)')
    parser.add_argument('-t', '--threshold-abs', type=float,
                        help='Segmentation threshold to use to separate bright cells from dark background')
    parser.add_argument('--only-segment', action='store_true',
                        help="Only segment frames, don't try to link them")

    parser.add_argument('image_dir', type=pathlib.Path,
                        help='Directory with individual frames of the time series to track')
    args = parser.parse_args()

    # Run the tracking algorithm and plot results
    tracker = OverlapTracker(args.image_dir,
                             threshold_abs=args.threshold_abs)
    tracker.load_frames()
    tracker.background_correct_frames()
    tracker.segment_frames()

    if args.only_segment:
        tracker.calculate_single_stats()
    else:
        tracker.link_frames()
        tracker.calculate_stats()
    tracker.plot_individual_frames(outdir=args.outdir)
