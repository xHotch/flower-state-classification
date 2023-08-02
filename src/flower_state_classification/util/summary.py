import json
import os

from flower_state_classification.util.timer import Timer


def create_summary(pipeline):
    source = pipeline.source
    frame_processor = pipeline.frame_processor
    debug_settings = pipeline.run_settings

    output_folder = debug_settings.output_folder
    detection_time = Timer.get_average_time("Detection") if Timer.has_timer("Detection") else "N/A"
    classification_time = Timer.get_average_time("Classification") if Timer.has_timer("Classification") else "N/A"
    # create summary file
    summary_json = os.path.join(output_folder, "summary.json")
    summary_dict = {
        "source": str(source),
        "nr_frames": source.get_framecount(),
        "nr_unique_detected_plants": len(frame_processor.classified_plants_new),
        "nr_frames_without_plants": len(frame_processor.frames_without_plants),
        "detector_model": str(frame_processor.detector),
        "average_detection_time": detection_time,
        "classifier_model": str(frame_processor.classifier),
        "average_classification_time": classification_time,
        "use_gpu": "TODO",
        "overall_time": Timer.get_average_time("Total Runtime"),
    }

    with open(summary_json, "w") as f:
        json.dump(summary_dict, f)
