import torch, torchvision, imageio, os, json, pandas
import imageio.v3 as iio
from PIL import Image
import random

DEBUG = False
# DEBUG = True

class DataProcessingPipeline:
    def __init__(self, operators=None):
        self.operators: list[DataProcessingOperator] = [] if operators is None else operators
        
    def __call__(self, data):
        for operator in self.operators:
            data = operator(data)
        return data
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline(self.operators + pipe.operators)



class DataProcessingOperator:
    def __call__(self, data):
        raise NotImplementedError("DataProcessingOperator cannot be called directly.")
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline([self]).__rshift__(pipe)


class DataProcessingOperatorRaw(DataProcessingOperator):
    def __call__(self, data):
        return data



class ToInt(DataProcessingOperator):
    def __call__(self, data):
        return int(data)


class ToFloat(DataProcessingOperator):
    def __call__(self, data):
        return float(data)


class ToStr(DataProcessingOperator):
    def __init__(self, none_value=""):
        self.none_value = none_value
    
    def __call__(self, data):
        if data is None: data = self.none_value
        return str(data)


class LoadImage(DataProcessingOperator):
    def __init__(self, convert_RGB=True):
        self.convert_RGB = convert_RGB
    
    def __call__(self, data: str):
        image = Image.open(data)
        if self.convert_RGB: image = image.convert("RGB")
        return image


class ImageCropAndResize(DataProcessingOperator):
    def __init__(self, height, width, max_pixels, height_division_factor, width_division_factor, min_pixels=None):
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        if min_pixels is not None and max_pixels is not None:
            assert min_pixels <= max_pixels, (
                f"min_pixels ({min_pixels}) must be <= max_pixels ({max_pixels})"
            )
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor

    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    def get_height_width(self, image):
        if self.height is None or self.width is None:
            width, height = image.size
            area = width * height
            if self.max_pixels and area > self.max_pixels:
                # Downscale to ~max_pixels (preserve aspect ratio).
                scale = (area / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            elif self.min_pixels and area < self.min_pixels:
                # Upscale to ~min_pixels (preserve aspect ratio).
                scale = (self.min_pixels / area) ** 0.5
                height, width = int(height * scale), int(width * scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        if DEBUG: print(self.height, self.width, "height: ", height, "width: ", width)
        return height, width
    
    
    def __call__(self, data: Image.Image):
        image = self.crop_and_resize(data, *self.get_height_width(data))
        return image


class ToList(DataProcessingOperator):
    def __call__(self, data):
        return [data]
    

class LoadVideo(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x,
                 rand_num_frames=False, min_num_frames=None, rand_num_frames_step=8):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor
        self.rand_num_frames = rand_num_frames
        # Build candidate list: min_num_frames, min+step, min+2*step, ..., num_frames
        # All candidates satisfy time_division constraint (num % factor == remainder)
        if rand_num_frames and min_num_frames is not None:
            self.rand_candidates = [
                n for n in range(min_num_frames, num_frames + 1, rand_num_frames_step)
                if n % time_division_factor == time_division_remainder
            ]
            if not self.rand_candidates:
                self.rand_candidates = [num_frames]
        else:
            self.rand_candidates = [num_frames]

    def sample_target_num_frames(self, video_len: int) -> int:
        """Randomly pick a target frame count given the actual video length.
        Used to pre-determine the frame count before loading src/tgt so both
        videos are loaded with the same number of frames."""
        if self.rand_num_frames:
            valid = [n for n in self.rand_candidates if n <= video_len]
            if valid:
                return random.choice(valid)
            else:
                target = video_len
                while target > 1 and target % self.time_division_factor != self.time_division_remainder:
                    target -= 1
                return target
        else:
            if video_len < self.num_frames:
                target = video_len
                while target > 1 and target % self.time_division_factor != self.time_division_remainder:
                    target -= 1
                return target
            return self.num_frames

    def get_num_frames(self, reader):
        video_len = int(reader.count_frames())
        # If a fixed target was pre-determined (set by __getitem__ to keep src/tgt in sync),
        # use it directly; otherwise sample on the fly.
        if hasattr(self, '_override_num_frames') and self._override_num_frames is not None:
            target = min(self._override_num_frames, video_len)
            # Ensure time_division constraint
            while target > 1 and target % self.time_division_factor != self.time_division_remainder:
                target -= 1
            return target
        return self.sample_target_num_frames(video_len)
        
    def __call__(self, data: str):
        reader = imageio.get_reader(data)
        num_frames = self.get_num_frames(reader)
        frames = []
        for frame_id in range(num_frames):
            frame = reader.get_data(frame_id)
            frame = Image.fromarray(frame)
            frame = self.frame_processor(frame)
            frames.append(frame)
        reader.close()
        return frames


class SequencialProcess(DataProcessingOperator):
    def __init__(self, operator=lambda x: x):
        self.operator = operator
        
    def __call__(self, data):
        return [self.operator(i) for i in data]


class LoadMaskVideo(DataProcessingOperator):
    """Load a mask video (mp4 / image) as a list of single-channel PIL images ('L' mode).

    Behaviour mirrors LoadVideo for frame-count sampling so the returned mask
    aligns frame-by-frame with the paired src/tgt videos. The optional
    `_override_num_frames` attribute (set by UnifiedDataset) is honoured to
    keep src/tgt/mask in sync within a single __getitem__ call.
    """
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1,
                 frame_processor=lambda x: x,
                 rand_num_frames=False, min_num_frames=None, rand_num_frames_step=8):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.frame_processor = frame_processor
        self.rand_num_frames = rand_num_frames
        if rand_num_frames and min_num_frames is not None:
            self.rand_candidates = [
                n for n in range(min_num_frames, num_frames + 1, rand_num_frames_step)
                if n % time_division_factor == time_division_remainder
            ]
            if not self.rand_candidates:
                self.rand_candidates = [num_frames]
        else:
            self.rand_candidates = [num_frames]

    def _resolve_num_frames(self, video_len: int) -> int:
        if hasattr(self, '_override_num_frames') and self._override_num_frames is not None:
            target = min(self._override_num_frames, video_len)
            while target > 1 and target % self.time_division_factor != self.time_division_remainder:
                target -= 1
            return target
        if self.rand_num_frames:
            valid = [n for n in self.rand_candidates if n <= video_len]
            if valid:
                return random.choice(valid)
        if video_len < self.num_frames:
            target = video_len
            while target > 1 and target % self.time_division_factor != self.time_division_remainder:
                target -= 1
            return target
        return self.num_frames

    def __call__(self, data: str):
        # Single-image mask (rare but possible) -> wrap into a one-frame list.
        ext = data.split(".")[-1].lower()
        if ext in ("jpg", "jpeg", "png", "webp", "bmp"):
            img = Image.open(data).convert("L")
            img = self.frame_processor(img)
            return [img]
        reader = imageio.get_reader(data)
        video_len = int(reader.count_frames())
        num_frames = self._resolve_num_frames(video_len)
        frames = []
        for frame_id in range(num_frames):
            frame = reader.get_data(frame_id)
            frame = Image.fromarray(frame).convert("L")
            frame = self.frame_processor(frame)
            frames.append(frame)
        reader.close()
        return frames


class LoadGIF(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor
        
    def get_num_frames(self, path):
        num_frames = self.num_frames
        images = iio.imread(path, mode="RGB")
        if len(images) < num_frames:
            num_frames = len(images)
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
        
    def __call__(self, data: str):
        num_frames = self.get_num_frames(data)
        frames = []
        images = iio.imread(data, mode="RGB")
        for img in images:
            frame = Image.fromarray(img)
            frame = self.frame_processor(frame)
            frames.append(frame)
            if len(frames) >= num_frames:
                break
        return frames
    


class RouteByExtensionName(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data: str):
        file_ext_name = data.split(".")[-1].lower()
        for ext_names, operator in self.operator_map:
            if ext_names is None or file_ext_name in ext_names:
                return operator(data)
        raise ValueError(f"Unsupported file: {data}")


class RouteByType(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data):
        for dtype, operator in self.operator_map:
            if dtype is None or isinstance(data, dtype):
                return operator(data)
        raise ValueError(f"Unsupported data: {data}")


class LoadTorchPickle(DataProcessingOperator):
    def __init__(self, map_location="cpu"):
        self.map_location = map_location
        
    def __call__(self, data):
        return torch.load(data, map_location=self.map_location, weights_only=False)


class ToAbsolutePath(DataProcessingOperator):
    def __init__(self, base_path=""):
        self.base_path = base_path
        
    def __call__(self, data):
        return os.path.join(self.base_path, data)


class LoadAudio(DataProcessingOperator):
    def __init__(self, sr=16000):
        self.sr = sr
    def __call__(self, data: str):
        import librosa
        input_audio, sample_rate = librosa.load(data, sr=self.sr)
        return input_audio


class UnifiedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        repeat=1,
        data_file_keys=tuple(),
        main_data_operator=lambda x: x,
        special_operator_map=None,
        load_video_op=None,
        mask_keys=tuple(),
        mask_operator=None,
        load_mask_op=None,
    ):
        self.base_path = base_path
        self.metadata_path = metadata_path
        self.repeat = repeat
        self.data_file_keys = data_file_keys
        self.main_data_operator = main_data_operator
        self.cached_data_operator = LoadTorchPickle()
        self.special_operator_map = {} if special_operator_map is None else special_operator_map
        self.load_video_op = load_video_op  # LoadVideo instance for pre-sampling num_frames
        # Mask handling: opt-in via mask_keys. Mask cells may be empty/NaN -> None.
        self.mask_keys = tuple(mask_keys) if mask_keys else tuple()
        self.mask_operator = mask_operator
        self.load_mask_op = load_mask_op  # LoadMaskVideo instance for frame-sync override
        self.data = []
        self.cached_data = []
        self.load_from_cache = metadata_path is None
        self.load_metadata(metadata_path)
    
    @staticmethod
    def default_image_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        min_pixels=None,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor, min_pixels=min_pixels)),
            (list, SequencialProcess(ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor, min_pixels=min_pixels))),
        ])
    
    @staticmethod
    def default_video_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        num_frames=81, time_division_factor=4, time_division_remainder=1,
        rand_num_frames=False, min_num_frames=None, rand_num_frames_step=8,
        min_pixels=None,
        img_min_pixels=None,
    ):
        # min_pixels only applies to real video files (mp4/avi/...). Image/gif
        # branches keep the original behavior (no upscale).
        load_video_op = LoadVideo(
            num_frames, time_division_factor, time_division_remainder,
            frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor, min_pixels=min_pixels),
            rand_num_frames=rand_num_frames,
            min_num_frames=min_num_frames,
            rand_num_frames_step=rand_num_frames_step,
        )
        operator = RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> RouteByExtensionName(operator_map=[
                (("jpg", "jpeg", "png", "webp"), LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor, min_pixels=img_min_pixels) >> ToList()),
                (("gif",), LoadGIF(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                )),
                (("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"), load_video_op),
            ])),
        ])
        return operator, load_video_op

    @staticmethod
    def default_mask_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        num_frames=81, time_division_factor=4, time_division_remainder=1,
        rand_num_frames=False, min_num_frames=None, rand_num_frames_step=8,
        min_pixels=None,
    ):
        """Build a mask loader operator that mirrors `default_video_operator`
        but loads single-channel (grayscale) frames. Returns (operator, load_mask_op)
        so the caller can pass `load_mask_op` to UnifiedDataset for frame-count sync.
        """
        load_mask_op = LoadMaskVideo(
            num_frames, time_division_factor, time_division_remainder,
            frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor, min_pixels=min_pixels),
            rand_num_frames=rand_num_frames,
            min_num_frames=min_num_frames,
            rand_num_frames_step=rand_num_frames_step,
        )
        operator = RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> RouteByExtensionName(operator_map=[
                (("jpg", "jpeg", "png", "webp", "bmp"),
                 LoadImage(convert_RGB=False)
                 >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor, min_pixels=min_pixels)
                 >> ToList()),
                (("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm", "gif"), load_mask_op),
            ])),
        ])
        return operator, load_mask_op
        
    def search_for_cached_data_files(self, path):
        for file_name in os.listdir(path):
            subpath = os.path.join(path, file_name)
            if os.path.isdir(subpath):
                self.search_for_cached_data_files(subpath)
            elif subpath.endswith(".pth"):
                self.cached_data.append(subpath)
    
    def load_metadata(self, metadata_path):
        if metadata_path is None:
            print("No metadata_path. Searching for cached data files.")
            self.search_for_cached_data_files(self.base_path)
            print(f"{len(self.cached_data)} cached data files found.")
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        elif metadata_path.endswith(".jsonl"):
            metadata = []
            with open(metadata_path, 'r') as f:
                for line in f:
                    metadata.append(json.loads(line.strip()))
            self.data = metadata
        else:
            metadata = pandas.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]

    def __getitem__(self, data_id):
        max_retry = 30
        retry_count = 0
        while retry_count < max_retry:
            try:
                if self.load_from_cache:
                    data = self.cached_data[data_id % len(self.cached_data)]
                    data = self.cached_data_operator(data)
                else:
                    data = self.data[data_id % len(self.data)].copy()
                    src_key = data['src_video']
                    tgt_key = data['tgt_video']
                    # Pre-load frame count check for video pairs (cheap metadata read)
                    frame_err = self.check_frame_num(src_key, tgt_key)
                    if frame_err:
                        raise ValueError(frame_err)
                    # Pre-sample a single target num_frames so src and tgt are loaded
                    # with the same frame count (avoids mismatch when rand_num_frames=True)
                    if self.load_video_op is not None and self.load_video_op.rand_num_frames:
                        src_video_len = self._get_raw_frame_count(src_key)
                        if src_video_len > 0:
                            sampled = self.load_video_op.sample_target_num_frames(src_video_len)
                            self.load_video_op._override_num_frames = sampled
                        else:
                            self.load_video_op._override_num_frames = None
                    # Sync mask loader frame-count with the video loader so
                    # src/tgt/mask all share the same chosen num_frames.
                    if self.load_mask_op is not None and self.load_video_op is not None:
                        self.load_mask_op._override_num_frames = getattr(
                            self.load_video_op, '_override_num_frames', None)
                    for key in self.data_file_keys:
                        if key in data:
                            if key in self.mask_keys:
                                # Mask key: empty/NaN -> None; otherwise use mask_operator.
                                if self._is_blank(data[key]) or self.mask_operator is None:
                                    data[key] = None
                                else:
                                    data[key] = self.mask_operator(data[key])
                            elif key in self.special_operator_map:
                                data[key] = self.special_operator_map[key](data[key])
                            elif key in self.data_file_keys:
                                if isinstance(data[key], list):
                                    data[key] = [self.main_data_operator(item)[0] for item in data[key]]
                                else:
                                    data[key] = self.main_data_operator(data[key])
                err_message, data['src_video'] = self.check_paired_size(data['src_video'], data['tgt_video'])
                if err_message:
                    raise ValueError(err_message)
                # Validate mask alignment vs tgt (frame count + spatial size).
                for mkey in self.mask_keys:
                    mval = data.get(mkey, None)
                    if mval is None:
                        continue
                    merr = self._check_mask_alignment(mval, data['tgt_video'], mkey)
                    if merr:
                        raise ValueError(merr)
                if self.load_video_op is not None:
                    self.load_video_op._override_num_frames = None  # reset after successful load
                if self.load_mask_op is not None:
                    self.load_mask_op._override_num_frames = None
                return data
            except Exception as e:
                if self.load_video_op is not None:
                    self.load_video_op._override_num_frames = None  # always reset on error
                if self.load_mask_op is not None:
                    self.load_mask_op._override_num_frames = None
                err_str = str(e)
                _is_frame_mismatch = "frame count mismatch" in err_str or "mismatch frame length" in err_str
                if _is_frame_mismatch:
                    if not hasattr(self, '_frame_mismatch_count'):
                        self._frame_mismatch_count = 0
                    self._frame_mismatch_count += 1
                    if self._frame_mismatch_count % 1000 == 1:
                        print(f"[frame mismatch suppressed, count={self._frame_mismatch_count}] {err_str[:120]}")
                else:
                    print(f"Error {retry_count}/{max_retry} loading data {data_id} {src_key} {tgt_key}: {e}")
                retry_count += 1
                data_id = random.randint(0, len(self.data) - 1) % len(self.data)
                continue
        raise ValueError(f"Failed to load data {data_id} after {max_retry} retries.")

    def __len__(self):
        if self.load_from_cache:
            return len(self.cached_data) * self.repeat
        else:
            return len(self.data) * self.repeat
        
    def check_data_equal(self, data1, data2):
        # Debug only
        if len(data1) != len(data2):
            return False
        for k in data1:
            if data1[k] != data2[k]:
                return False
        return True
    
    VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "wmv", "mkv", "flv", "webm", "gif"}

    @staticmethod
    def _get_raw_frame_count(path: str) -> int:
        """Read frame count from video/GIF using cv2.VideoCapture (same as concat_side_by_side)."""
        try:
            import cv2
            cap = cv2.VideoCapture(path)
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            return n if n > 0 else -1
        except Exception:
            return -1

    def check_frame_num(self, src_path: str, tgt_path: str) -> str:
        """Pre-load check: compare raw frame counts of two video files.
        Returns an error message string, or empty string if OK.
        Only applies when both files are video/GIF; skips images.
        """
        src_ext = src_path.split(".")[-1].lower()
        tgt_ext = tgt_path.split(".")[-1].lower()
        if src_ext not in self.VIDEO_EXTENSIONS or tgt_ext not in self.VIDEO_EXTENSIONS:
            return ""
        src_n = self._get_raw_frame_count(src_path)
        tgt_n = self._get_raw_frame_count(tgt_path)
        if src_n < 0 or tgt_n < 0:
            return ""  # cannot determine, let downstream handle it
        if src_n != tgt_n:
            return f"frame count mismatch before loading: src={src_n} tgt={tgt_n} ({src_path})"
        return ""

    @staticmethod
    def _is_blank(v):
        """True if the metadata cell should be treated as 'no mask' (NaN / empty / 'nan')."""
        if v is None:
            return True
        try:
            # pandas NaN check without importing pandas here
            if isinstance(v, float) and v != v:
                return True
        except Exception:
            pass
        if isinstance(v, str):
            s = v.strip()
            return s == "" or s.lower() == "nan"
        return False

    def _check_mask_alignment(self, mask_frames, tgt_frames, mkey):
        """Mask must have same frame count and spatial size as tgt video.
        Returns "" on success, error message string otherwise.
        """
        if not isinstance(mask_frames, list) or not isinstance(tgt_frames, list):
            return ""
        if len(mask_frames) != len(tgt_frames):
            return f"mask frame count mismatch ({mkey}): mask={len(mask_frames)} tgt={len(tgt_frames)}"
        if len(mask_frames) == 0 or len(tgt_frames) == 0:
            return ""
        mw, mh = mask_frames[0].size
        tw, th = tgt_frames[0].size
        if mw != tw or mh != th:
            return f"mask spatial mismatch ({mkey}): mask={mw}x{mh} tgt={tw}x{th}"
        return ""

    def check_paired_size(self, data1, data2, aspect_ratio_threshold=0.15):
        err_message = ""
        if len(data1) != len(data2):
            err_message += f'mismatch frame length {len(data1)} {len(data2)}'
            return err_message, data1

        src_w, src_h = data1[0].size
        tgt_w, tgt_h = data2[0].size
        if src_w != tgt_w or src_h != tgt_h:
            is_image_pair = len(data1) == 1 and len(data2) == 1
            if is_image_pair:
                src_ratio = src_w / src_h
                tgt_ratio = tgt_w / tgt_h
                ratio_diff = abs(src_ratio - tgt_ratio)
                if ratio_diff <= aspect_ratio_threshold:
                    data1 = [frame.resize((tgt_w, tgt_h), Image.BILINEAR) for frame in data1]
                else:
                    err_message += (
                        f'image aspect ratio mismatch: src {src_w}x{src_h} (ratio={src_ratio:.4f}) '
                        f'tgt {tgt_w}x{tgt_h} (ratio={tgt_ratio:.4f}) diff={ratio_diff:.4f}'
                    )
            else:
                if src_w != tgt_w:
                    err_message += f'mismatch width size {src_w} {tgt_w}'
                if src_h != tgt_h:
                    err_message += f'mismatch height size {src_h} {tgt_h}'
        return err_message, data1
