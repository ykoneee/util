import concurrent.futures
import itertools
import os
import re
import shutil
import zipfile
from pathlib import Path

import numpy as np
import zarr
from tqdm import tqdm

from dataset_manager import ZARR_DATASET_CREATER

zarr.blosc.use_threads = False
from csi_music_utils import write_npy_in_zip, load_npy_from_bytes
from intel5300dat_process import get_csi_from_bytes
from math_utils import split_array_bychunk, complex_to_amp_pha
from widar_reinpl_series import gen_fir_filter

pattern = re.compile(r"user(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-r(\d+).dat")
year_num_pattern = re.compile(r"(2018\d{4})")
widar3_env_label = {
    "20181109": 1,
    "20181112": 1,
    "20181115": 1,
    "20181116": 1,
    "20181117": 2,
    "20181118": 2,
    "20181121": 1,
    "20181127": 2,
    "20181128": 2,
    "20181130": 1,
    "20181204": 2,
    "20181205": 2,
    "20181208": 2,
    "20181209": 2,
    "20181211": 3,
}

widar3_gesture_mapping = {
    "20181109": [0, 1, 2, 3, 9, 10],
    "20181112": [12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
    "20181115": [0, 1, 2, 9, 10, 11],
    "20181116": [12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
    "20181117": [0, 1, 2, 9, 10, 11],
    "20181118": [0, 1, 2, 9, 10, 11],
    "20181121": [3, 4, 5, 6, 7, 8],
    "20181127": [3, 4, 5, 6, 7, 8],
    "20181128": [0, 1, 2, 4, 5, 8],
    "20181130": [0, 1, 2, 3, 4, 5, 6, 7, 8],
    "20181204": [0, 1, 2, 3, 4, 5, 6, 7, 8],
    "20181205": [3, 4, 5, 6, 7, 8],
    "20181208": [0, 1, 2, 3],
    "20181209": [0, 1, 2, 3, 5, 8],
    "20181211": [0, 1, 2, 3, 5, 8],
}
widar3_gesture_mapping = {k: np.array(v) + 1 for k, v in widar3_gesture_mapping.items()}


def widar3_infor(p):
    root_path = Path(p)
    for i, x1 in enumerate(
            itertools.chain.from_iterable(
                [
                    root_path.glob("20181109.zip"),
                    root_path.glob("20181115.zip"),
                    root_path.glob("20181117.zip"),
                    root_path.glob("20181118.zip"),
                    root_path.glob("20181121.zip"),
                    root_path.glob("20181127.zip"),
                    root_path.glob("20181128.zip"),
                    root_path.glob("20181130*.zip"),
                    root_path.glob("20181204.zip"),
                    root_path.glob("20181205.zip"),
                    root_path.glob("20181208.zip"),
                    root_path.glob("20181209.zip"),
                    root_path.glob("20181211.zip"),
                ]
            )
    ):
        print(x1)
        archive = zipfile.ZipFile(x1, "r")
        l = []
        for x2 in archive.infolist():
            # print(x2.filename)
            if "yumeng" in x2.filename or "baidu" in x2.filename:
                print(x2)
                continue
            if not x2.is_dir():
                n = (x2.filename).split("/")[-1]
                try:
                    (
                        userid,
                        gesture,
                        location,
                        face_orientation,
                        sampleid,
                        receiverid,
                    ) = map(int, re.findall(pattern, n)[0])
                    l.append(
                        (
                            userid,
                            gesture,
                            location,
                            face_orientation,
                            sampleid,
                            receiverid,
                        )
                    )
                except BaseException as e:
                    print("Error:", x1, x2, n)
                    print(e)
        l = np.array(
            l,
            dtype=[
                ("userid", "i4"),
                ("gesture", "i4"),
                ("location", "i4"),
                ("face_orientation", "i4"),
                ("sampleid", "i4"),
                ("receiverid", "i4"),
            ],
        )
        print(np.unique(l["location"]))


def widar3_iter_process(extract_condition, in_path, out_path, iter_f):
    global zip_file_info_list_chunked, process_idx_total, mask_total, write_idx, zip_name, zip_raw, zip_amp, zip_pha
    cpu_rate = 2
    process_num = int(os.cpu_count() // cpu_rate)
    root_path = Path(in_path)
    out_path = Path(out_path).resolve()
    if out_path.exists():
        shutil.rmtree(out_path)
    out_path.mkdir()
    write_idx = 0
    multi_label_total = []
    mask_total = []
    process_idx_total = []
    global filter_fir
    filter_fir = gen_fir_filter(1000, 250)
    # filter_fir = None
    with zipfile.ZipFile(
            out_path / "raw.zip", "w", zipfile.ZIP_STORED
    ) as zip_raw, zipfile.ZipFile(
        out_path / "amp.zip", "w", zipfile.ZIP_STORED
    ) as zip_amp, zipfile.ZipFile(
        out_path / "pha.zip", "w", zipfile.ZIP_STORED
    ) as zip_pha:
        for zip_idx, zip_name in enumerate(
                itertools.chain.from_iterable(
                    [
                        # root_path.glob("20181109.zip"),
                        ## root_path.glob("20181115.zip"),
                        # root_path.glob("20181117.zip"),
                        # root_path.glob("20181118.zip"),
                        # root_path.glob("20181121.zip"),
                        # root_path.glob("20181127.zip"),
                        # root_path.glob("20181128.zip"),
                        root_path.glob("20181130*.zip"),
                        root_path.glob("20181204.zip"),
                        root_path.glob("20181205.zip"),
                        root_path.glob("20181208.zip"),
                        root_path.glob("20181209.zip"),
                        root_path.glob("20181211.zip"),
                    ]
                )
        ):
            print(zip_name)
            print(f"len of data {sum(mask_total)}")
            year_num = re.findall(year_num_pattern, zip_name.name)[0]
            roomid = widar3_env_label[year_num]
            with zipfile.ZipFile(zip_name, "r") as zip_f:
                zip_file_info_list = zip_f.infolist()
                zip_file_info_list_selected = []
                csi_info_list = []
                for zip_file_info in zip_file_info_list:
                    if (
                            "yumeng" in zip_file_info.filename
                            or "baidu" in zip_file_info.filename
                    ):
                        continue
                    if not zip_file_info.is_dir():
                        file_name = (zip_file_info.filename).split("/")[-1]

                        (
                            userid,
                            gesture,
                            location,
                            face_orientation,
                            sampleid,
                            receiverid,
                        ) = map(int, re.findall(pattern, file_name)[0])
                        gesture = widar3_gesture_mapping[year_num][gesture - 1]

                        if (
                                True
                                and gesture in extract_condition["gesture"]
                                and receiverid in extract_condition["receiverid"]
                                and face_orientation
                                in extract_condition["face_orientation"]
                                and location in extract_condition["location"]
                                # and sampleid == 1
                        ):
                            zip_file_info_list_selected.append(zip_file_info.filename)
                            csi_info_list.append(
                                (
                                    roomid,
                                    userid,
                                    gesture,
                                    location,
                                    face_orientation,
                                    sampleid,
                                    receiverid,
                                )
                            )

                multi_label_total.extend(csi_info_list)
                del zip_file_info_list
                zip_file_info_list_chunked = split_array_bychunk(
                    zip_file_info_list_selected, 14
                )
            iter_f()
    multi_label_total = np.array(
        multi_label_total,
        dtype=[
            ("roomid", "i4"),
            ("userid", "i4"),
            ("gesture", "i4"),
            ("location", "i4"),
            ("face_orientation", "i4"),
            ("sampleid", "i4"),
            ("receiverid", "i4"),
        ],
    )
    mask_total = np.array(mask_total)
    process_idx_total = np.array(process_idx_total)
    rm_info_all = multi_label_total[~mask_total]
    multi_label_total = multi_label_total[mask_total]
    np.save(out_path / "multi_label", multi_label_total)
    np.save(out_path / "rm_info_all", rm_info_all)
    # np.save(out_path / "process_idx_total", process_idx_total)


def process_zip_info_list(zip_file_info_list, zip_path, process_idx):
    global filter_fir
    mask = []
    array_l = []
    with zipfile.ZipFile(zip_path, "r") as zip_f:
        for zip_file_info in zip_file_info_list:
            b_zip_dat = zip_f.read(zip_file_info)
            raw_csi = get_csi_from_bytes(b_zip_dat)
            del b_zip_dat
            if len(raw_csi) < 900:
                mask.append(False)
                continue
            else:
                mask.append(True)
            # raw_csi = apply_csi_filter(raw_csi, filter_fir)
            # raw_csi, _ = conj_multi_csi(raw_csi)
            amp_csi, pha_csi = complex_to_amp_pha(raw_csi)
            array_l.append([raw_csi, amp_csi, pha_csi])
            # array_l.append(raw_csi)

    return mask, array_l, process_idx


def widar3_zip2np_subf():
    global zip_file_info_list_chunked, process_idx_total, mask_total, write_idx, zip_name, zip_raw, zip_amp, zip_pha
    with concurrent.futures.ProcessPoolExecutor(5) as executor:
        for concurrent_ret in tqdm(
                executor.map(
                    process_zip_info_list,
                    zip_file_info_list_chunked,
                    itertools.repeat(zip_name),
                    range(len(zip_file_info_list_chunked)),
                ),
                total=len(zip_file_info_list_chunked),
        ):
            mask, array_l, idx = concurrent_ret
            process_idx_total.append(idx)
            mask_total.extend(mask)
            for raw, amp, pha in array_l:
                write_npy_in_zip(f"{write_idx}", raw, zip_raw)
                write_npy_in_zip(f"{write_idx}", amp, zip_amp)
                write_npy_in_zip(f"{write_idx}", pha, zip_pha)
                write_idx += 1
                del raw, amp, pha

        executor.shutdown()


def widar3_zip2np_sequence_subf():
    global zip_file_info_list_chunked, process_idx_total, mask_total, write_idx, zip_name, zip_raw, zip_amp, zip_pha
    for concurrent_in in tqdm(
            zip_file_info_list_chunked,
            total=len(zip_file_info_list_chunked),
    ):
        concurrent_ret = process_zip_info_list(concurrent_in, zip_name, 0)
        mask, array_l, idx = concurrent_ret
        process_idx_total.append(idx)
        mask_total.extend(mask)
        for raw, amp, pha in array_l:
            write_npy_in_zip(f"{write_idx}", raw, zip_raw)
            write_npy_in_zip(f"{write_idx}", amp, zip_amp)
            write_npy_in_zip(f"{write_idx}", pha, zip_pha)
            write_idx += 1
            del raw, amp, pha


def extract_widar_data():
    extract_condition = {
        "receiverid": (3,),
        "face_orientation": (
            1,
            2,
            3,
        ),
        "location": (
            1,
            2,
            5,
        ),
        "gesture": (1, 2, 3, 4, 6, 9),
    }
    # root = Path("/media/yk/Samsung_T5/")
    root = Path("/media/yk/D4FA828F299D817A/")
    zip_path = root / "Widar3.0ReleaseData/CSI"
    np_path = root / "Widar3.0ReleaseData/np_f"
    # widar3_zip2np(extract_condition, in_path=zip_path, out_path=np_path)
    widar3_iter_process(
        extract_condition,
        in_path=zip_path,
        out_path=np_path,
        # iter_f=widar3_zip2np_sequence_subf,
        iter_f=widar3_zip2np_subf,
    )


def test_widar_data():
    # root = Path("/media/yk/Samsung_T5/Widar3.0ReleaseData/CSI")
    root = Path("/media/yk/D4FA828F299D817A/Widar3.0ReleaseData/CSI")
    widar3_infor(
        root,
    )


def load_widar_data():
    root = Path("/media/yk/D4FA828F299D817A/")
    # root = Path("/media/yk/Samsung_T5/")
    np_path = root / "Widar3.0ReleaseData/np_f"
    out_path = root / "Widar3.0ReleaseData/zarr_f"

    multi_label = np.load(np_path / "multi_label.npy")
    z_c = ZARR_DATASET_CREATER(out_path, None, 8000)
    with zipfile.ZipFile(np_path / "raw.zip", "r") as zip_raw, zipfile.ZipFile(
            np_path / "amp.zip", "r"
    ) as zip_amp, zipfile.ZipFile(np_path / "pha.zip", "r") as zip_pha:
        infol = zip_raw.infolist()
        for i, x in tqdm(enumerate(infol), total=len(infol)):
            n = x.filename
            raw = load_npy_from_bytes(zip_raw.read(n))
            amp = load_npy_from_bytes(zip_amp.read(n))
            pha = load_npy_from_bytes(zip_pha.read(n))
            act = multi_label[i]["gesture"]
            env = multi_label[i]["roomid"]
            loc = multi_label[i]["location"]
            user = multi_label[i]["userid"]

            raw = raw.astype(np.complex64)
            z_c.append(
                csi_data_raw=raw,
                csi_data_amp=amp,
                csi_data_pha=pha,
                csi_label_act=act,
                csi_label_env=env,
                csi_label_user=user,
                csi_label_loc=loc,
            )
        z_c.close()

    # f= zipfile.ZipFile(np_path / "raw.zip", "r", zipfile.ZIP_STORED)
    # zip_raw.read('0')
    # f.close()


if __name__ == "__main__":
    # test_widar_data()
    # extract_widar_data()
    load_widar_data()
