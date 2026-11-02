def convert(job):
    (source, output, model, hierarchy, visual, sigma, frequency) = job
    from .solver import retarget_file
    from .bvh import Bvh
    from .csv2bvh import csv_to_bvh
    from .smooth_motion import gaussian_smooth_motion, load_csv_motion, save_csv_motion

    csv_path = output / (source.stem + ".csv")
    retarget_file(source, csv_path, model_path=model, visual=visual, max_frequency=frequency)
    bvh = Bvh()
    bvh.parse_file(source)
    if sigma > 0:
        (header, motion, indices) = load_csv_motion(csv_path)
        save_csv_motion(csv_path, header, gaussian_smooth_motion(motion, sigma), indices)
    csv_to_bvh(hierarchy, csv_path, csv_path.with_suffix(".bvh"), fps=bvh.fps)
    return str(csv_path)
