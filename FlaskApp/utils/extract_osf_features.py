import parselmouth
import numpy as np

def extract_osf_features(wav_path, gender=None):
    snd = parselmouth.Sound(wav_path)
    results = {}

    try:
        # --- Formant F1 ---
        formant_ceiling = 5500.0 if gender == "female" else 5000.0
        formant = snd.to_formant_burg(time_step=0.01, max_number_of_formants=5,
                                      maximum_formant=formant_ceiling, window_length=0.025, pre_emphasis_from=50.0)
        time_points = np.linspace(0, snd.duration, 100)
        f1_values = [formant.get_value_at_time(1, t) for t in time_points]
        results["avg_F1"] = np.nanmean(f1_values)
    except Exception as e:
        print("F1 extraction failed:", e)
        results["avg_F1"] = np.nan

    try:
        # --- Pitch features ---
        pitch_floor = 75 if gender == "male" else 100
        pitch_ceiling = 300 if gender == "male" else 500
        time_step = 0.01  # 10ms = good balance of precision and performance
        pitch = snd.to_pitch_ac(time_step=time_step, pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling)        
        results["mean_F0"] = parselmouth.praat.call(pitch, "Get mean", 0, 0, "Hertz")
        results["sd_F0"] = parselmouth.praat.call(pitch, "Get standard deviation", 0, 0, "Hertz")
    except Exception as e:
        print("Pitch extraction failed:", e)

    try:
      point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", pitch_floor, pitch_ceiling)
    
      results["jitter_s"] = parselmouth.praat.call(
        point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)

      results["jitter_local_absolute"] = parselmouth.praat.call(
        point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)

      results["shimmer"] = parselmouth.praat.call(
        [snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

      results["shimmer_local_dB"] = parselmouth.praat.call(
        [snd, point_process], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    except Exception as e:
      print("❌ Jitter/Shimmer extraction failed:", e)

    try:
        harmonicity = snd.to_harmonicity_cc()
        results["mean_HNR"] = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)
    except Exception as e:
        print("HNR extraction failed:", e)

    return results
