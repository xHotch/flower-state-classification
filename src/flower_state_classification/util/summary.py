import json
import os


def create_summary(pipeline):
    source = pipeline.source
    frame_processor = pipeline.frame_processor
    debug_settings = pipeline.debug_settings

    output_folder = debug_settings.output_folder

    #create summary file
    summary_json = os.path.join(output_folder, "summary.json")
    summary_dict = {
        "source": str(source),
        "nr_frames": source.get_framecount(),
        "nr_detected_plants": len(frame_processor.detected_plants),
        "nr_frames_without_plants": len(frame_processor.frames_without_plants),
        "detector_model": str(frame_processor.detector),
        "classifier_model": str(frame_processor.classifier),
        "use_gpu": frame_processor.use_gpu,
        "overall_time": pipeline.run.total_time
    }

    with open(summary_json, "w") as f:
        json.dump(summary_dict, f)
