import json
import os

from flower_state_classification.util.timer import Timer



def create_summary(pipeline):
    source = pipeline.source
    frame_processor = pipeline.frame_processor
    debug_settings = pipeline.run_settings

    output_folder = debug_settings.output_folder

    #create summary file
    summary_json = os.path.join(output_folder, "summary.json")
    summary_dict = {
        "source": str(source),
        "nr_frames": source.get_framecount(),
        "nr_detected_plants": sum(len(lst) for lst in frame_processor.classified_plants_new.values()),
        "nr_frames_without_plants": len(frame_processor.frames_without_plants),
        "detector_model": str(frame_processor.detector),
        "average_detection_time": Timer.get_average_time("Detection"),
        "classifier_model": str(frame_processor.classifier),
        "average_classification_time": Timer.get_average_time("Classification"),
        "use_gpu": frame_processor.use_gpu,
        "overall_time": pipeline.run.total_time
    }

    with open(summary_json, "w") as f:
        json.dump(summary_dict, f)
